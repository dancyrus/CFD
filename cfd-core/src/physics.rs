//! Walls, boundary conditions, sponge, carbuncle sensor, initialization,
//! geometry flips. **Session B owns this file and nothing else.** Signatures
//! are frozen — they are what `step.rs` and session A's sweeps call. If you
//! believe one is wrong, print `CONTRACT CHANGE REQUEST:` with the exact diff
//! and stop.
//!
//! Layout: all padded slices are `g.glen()` long, indexed with `Grid::gidx`
//! (see `lib.rs`). Every reduction accumulates in f64. All state is
//! non-dimensional, chamber-referenced: R = 1, p = rho*T, chamber (1, 0, 0, 1).

use std::sync::atomic::{AtomicBool, Ordering};

use cfd_contract::kernels::{cons_to_prim, prim_to_cons};
use cfd_contract::{
    Ambient, Chamber, Cons, GasModel, Grid, Numerics, Prim, Real, SolidField, WallMode, NG,
    P_MIN_ABS,
};

/// Wall flux at a fluid/solid face. `w` is the fluid cell's CANONICAL
/// primitives. `sgn` is +1 when the fluid is on the low side of the face,
/// -1 on the high side. Returns [0, sgn*ps, 0, 0] — mass and energy flux are
/// bit-exactly zero. Special-cased, NOT computed by calling hllc_flux with a
/// mirrored state (the S_M = 0 cancellation is exact in real arithmetic but
/// not under FMA contraction). See docs/physics-reference.md §3.
///
/// CONSUMPTION RULE (session A): the sgn premultiplication encodes the face
/// orientation so the sweep applies the UNIFORM rule `rhs[fluid] -= F/dz`
/// (r: `-= r_face*F/(dr*r_c)`) for a wall face on EITHER side of the cell.
/// Plugging this return directly into the textbook difference as F_{i-1/2}
/// for a low-side wall flips the sign and breaks quiescent equilibrium —
/// the well_balanced acceptance test catches exactly that.
pub fn wall_flux_z(w: Prim, sgn: Real, gamma: Real) -> Cons {
    [0.0, sgn * wall_star_pressure(w[0], w[1], w[3], sgn, gamma), 0.0, 0.0]
}

/// Returns [0, 0, sgn*ps, 0].
pub fn wall_flux_r(w: Prim, sgn: Real, gamma: Real) -> Cons {
    [0.0, 0.0, sgn * wall_star_pressure(w[0], w[2], w[3], sgn, gamma), 0.0]
}

/// HLLC star pressure of the mirrored Riemann problem at S_M = 0, Davis speed.
/// For flow into the wall (un > 0): ps = p + rho*un*(2*un + a) > p. Receding:
/// ps = p - rho*|un|*a, the acoustic expansion, floored at P_MIN_ABS.
#[inline]
fn wall_star_pressure(rho: Real, u: Real, p: Real, sgn: Real, gamma: Real) -> Real {
    let a = (gamma * p / rho).sqrt();
    let un = sgn * u; // outward-from-fluid normal velocity
    let sl = -(un.abs() + a);
    (p + rho * un * (un - sl)).max(P_MIN_ABS)
}

/// Subsonic stagnation inlet ghost state (NASA/TM-2011-217181 construction):
/// R⁻ extrapolated from the interior, stagnation relation closes the system.
/// Non-dimensional: a0² = gamma*t0. The `D = max(D, 0)` clamp is MANDATORY —
/// D goes negative under transient reverse flow and NaNs before the first
/// frame otherwise. v_b = 0: axial injection, tangential velocity defined.
#[inline]
fn inlet_ghost(wi: Prim, chamber: &Chamber, g: Real) -> Cons {
    let a02 = g * chamber.t0;
    let ai = (g * wi[3] / wi[0]).sqrt();
    let rm = wi[1] - 2.0 * ai / (g - 1.0);
    let aa = (g + 1.0) / (g - 1.0);
    let d = ((g + 1.0) * a02 / (g - 1.0) - 0.5 * (g - 1.0) * rm * rm).max(0.0);
    // The 1e-6 floor is a guard beyond the reference: if D clamps to 0 while
    // R- > 0 (supersonic reverse inflow), a_b would be <= 0 and rho_b = 0/0.
    let ab = ((-rm + d.sqrt()) / aa).max(1e-6);
    let ub = (rm + 2.0 * ab / (g - 1.0)).max(0.0);
    let pb = chamber.p0 * (ab * ab / a02).powf(g / (g - 1.0));
    let rb = g * pb / (ab * ab);
    prim_to_cons([rb, ub, 0.0, pb], g)
}

/// Downstream outflow ghost, the same three-way split `farfield_ghost` uses on
/// the radial normal. Outgoing supersonic (u >= a): copy the interior cell —
/// all four characteristics exit, exact and non-reflecting. Outgoing subsonic
/// (0 <= u < a): impose p_a, extrapolate entropy and R⁺. Reversed (u < 0): the
/// ambient reservoir at rest — incoming gas carries AMBIENT entropy and
/// AMBIENT tangential velocity, never the interior's, which is downstream of
/// the face and says nothing about what enters through it.
///
/// The reversal branch and the SIGNED supersonic test are both load-bearing
/// (A2). Without them, u < 0 fell into the subsonic construction, which
/// extrapolates interior entropy and v along characteristics that are entering
/// the domain — measured: interior u = -0.30 produced a ghost with u = +1.45,
/// a spurious outward jet at M 2.5. And `|u| >= a` read supersonic INFLOW as
/// supersonic outflow, copying the interior bit-exactly: zero conditions on
/// four incoming characteristics, ill-posed — a sustained M 4.49 inflow grew
/// domain mass 41.7% over 3000 steps with nothing constraining it.
#[inline]
fn outflow_ghost(ui: Cons, wi: Prim, ambient: &Ambient, g: Real) -> Cons {
    let un = wi[1];
    if un < 0.0 {
        let rho_a = ambient.p / ambient.t;
        return prim_to_cons([rho_a, 0.0, 0.0, ambient.p], g);
    }
    let ai = (g * wi[3] / wi[0]).sqrt();
    if un >= ai {
        return ui;
    }
    let rb = wi[0] * (ambient.p / wi[3]).powf(1.0 / g);
    let ab = (g * ambient.p / rb).sqrt();
    let ub = un + 2.0 * (ai - ab) / (g - 1.0);
    prim_to_cons([rb, ub, wi[2], ambient.p], g)
}

/// Radial far-field ghost: the outflow construction with the radial normal;
/// if flow is entering (v_i < 0), the ambient reservoir.
#[inline]
fn farfield_ghost(ui: Cons, wi: Prim, ambient: &Ambient, g: Real) -> Cons {
    let vi = wi[2];
    if vi < 0.0 {
        let rho_a = ambient.p / ambient.t;
        return prim_to_cons([rho_a, 0.0, 0.0, ambient.p], g);
    }
    let ai = (g * wi[3] / wi[0]).sqrt();
    if vi >= ai {
        return ui;
    }
    let rb = wi[0] * (ambient.p / wi[3]).powf(1.0 / g);
    let ab = (g * ambient.p / rb).sqrt();
    let vb = vi + 2.0 * (ai - ab) / (g - 1.0);
    prim_to_cons([rb, wi[1], vb, ambient.p], g)
}

/// Axis mirror (two ghost rows, (rho, u_z, -u_r, p) — applies at r-min in
/// Planar mode too), stagnation inlet on open z-min cells, supersonic/subsonic
/// outflow at z-max, radial far field at r-max. See docs/physics-reference.md
/// §4 — including the MANDATORY `D = max(D, 0)` clamp in the inlet.
pub fn fill_ghosts(u: &mut [Cons], g: &Grid, solid: &[bool], gas: &GasModel,
                   chamber: &Chamber, ambient: &Ambient, _n: &Numerics) {
    let gm = gas.gamma;
    let ng = NG as isize;
    let nz = g.nz as isize;
    let nr = g.nr as isize;

    // z boundaries first, interior rows only. Ghost cells behind a solid
    // boundary cell just copy it — the sweeps never form a fluid flux there.
    for ir in 0..nr {
        let i0 = g.gidx(0, ir);
        let ghost = if solid[i0] {
            u[i0]
        } else {
            inlet_ghost(cons_to_prim(u[i0], gm), chamber, gm)
        };
        u[g.gidx(-1, ir)] = ghost;
        u[g.gidx(-2, ir)] = ghost;

        let i1 = g.gidx(nz - 1, ir);
        let ghost = if solid[i1] {
            u[i1]
        } else {
            outflow_ghost(u[i1], cons_to_prim(u[i1], gm), ambient, gm)
        };
        u[g.gidx(nz, ir)] = ghost;
        u[g.gidx(nz + 1, ir)] = ghost;
    }

    // r boundaries second, full padded width, so the corners are filled from
    // the just-written z ghosts and every ghost cell holds a valid state.
    for iz in -ng..nz + ng {
        // Axis: ghost row -1-k mirrors interior row k (clamped for nr = 1).
        for k in 0..ng {
            let src = u[g.gidx(iz, k.min(nr - 1))];
            u[g.gidx(iz, -1 - k)] = [src[0], src[1], -src[2], src[3]];
        }
        let i1 = g.gidx(iz, nr - 1);
        let ghost = if solid[i1] {
            u[i1]
        } else {
            farfield_ghost(u[i1], cons_to_prim(u[i1], gm), ambient, gm)
        };
        u[g.gidx(iz, nr)] = ghost;
        u[g.gidx(iz, nr + 1)] = ghost;
    }

    if _n.wall_mode == WallMode::ColumnReflect {
        column_reflect_fill(u, g, solid);
    }
}

/// WallMode::ColumnReflect (abort-ladder rung 3, also used as a measurement of
/// the staircase wall's own contribution to the wall layer): per column, every
/// solid cell is overwritten with the mirror image of the fluid below the bore
/// (u_r negated), so the sweeps can run the REGULAR flux solver at wall faces
/// and the bore face becomes a reflecting condition. Only star-convex-in-r
/// geometry survives; solid cells stay frozen (accumulate skips them) and are
/// refilled here before every stage.
fn column_reflect_fill(u: &mut [Cons], g: &Grid, solid: &[bool]) {
    let ng = NG as isize;
    let nr = g.nr as isize;
    for iz in -ng..(g.nz as isize + ng) {
        let Some(b) = (0..nr).find(|&ir| solid[g.gidx(iz, ir)]) else { continue };
        for ir in b..nr + ng {
            let src = (2 * b - 1 - ir).max(0); // mirror about the bore face
            let mut m = u[g.gidx(iz, src)];
            m[2] = -m[2];
            u[g.gidx(iz, ir)] = m;
        }
    }
}

static SPONGE_ENTRY_WARNED: AtomicBool = AtomicBool::new(false);

/// dt-based, sigma_max = 12*a_ambient/L_sponge. NOT the per-step form (4-6x
/// too weak, resolution-dependent). Radial far-field rows only — never the
/// downstream boundary, where the core is supersonic. `cells` = 0 disables.
pub fn apply_sponge(u: &mut [Cons], g: &Grid, dt: Real, ambient: &Ambient,
                    gas: &GasModel, cells: usize) {
    if cells == 0 {
        return;
    }
    let cells = cells.min(g.nr);
    let gm = gas.gamma;
    let rho_a = ambient.p / ambient.t;
    let ua = prim_to_cons([rho_a, 0.0, 0.0, ambient.p], gm);
    let a_amb = (gm * ambient.t).sqrt(); // a² = gamma*p/rho = gamma*T
    let ir0 = g.nr - cells;
    // L_sponge is the PHYSICAL depth of the outer `cells` rows — on a graded
    // far field those rows are wide, and an index-based depth would misplace
    // the profile and mis-scale sigma_max.
    let l_sponge = (g.lr() as Real) - g.r_face(ir0);
    let sigma_max = 12.0 * a_amb / l_sponge; // 1/time

    // Diagnostic (once): the plume reaching the sponge entry means the far
    // field is too small and the damping is corrupting physics.
    if !SPONGE_ENTRY_WARNED.load(Ordering::Relaxed) {
        for iz in 0..g.nz {
            let w = cons_to_prim(u[g.gidx(iz as isize, ir0 as isize)], gm);
            if w[3] > 1.5 * ambient.p {
                SPONGE_ENTRY_WARNED.store(true, Ordering::Relaxed);
                eprintln!(
                    "cfd-core: plume reached the sponge entry (p = {:.3e} > 1.5*p_a = {:.3e}) \
                     — far-field damping may be corrupting it",
                    w[3], 1.5 * ambient.p
                );
                break;
            }
        }
    }

    for ir in ir0..g.nr {
        let s = (g.r_center(ir) - g.r_face(ir0)) / l_sponge;
        // .min(1.0) is a stability guard beyond the reference: an explicit
        // relaxation step must not overshoot past ambient. At CFL 0.4 the
        // coefficient is ~0.2, so the clamp never engages in normal runs.
        let coef = (dt * sigma_max * s * s).min(1.0);
        for iz in 0..g.nz {
            let c = &mut u[g.gidx(iz as isize, ir as isize)];
            for k in 0..4 {
                c[k] -= coef * (c[k] - ua[k]);
            }
        }
    }
}

/// Omega = min/max of p over a +/-2 cell axial window; where Omega < 0.7 set
/// mask on cells i-1..=i+1 of that row. The caller uses HLL for the RADIAL
/// fluxes of masked cells; axial fluxes always use HLLC.
pub fn carbuncle_mask(w: &[Prim], g: &Grid, mask: &mut [bool]) {
    mask.fill(false);
    let nz = g.nz as isize;
    for ir in 0..g.nr as isize {
        for iz in 0..nz {
            let mut pmin = Real::INFINITY;
            let mut pmax = 0.0 as Real;
            for d in -2..=2 {
                let p = w[g.gidx(iz + d, ir)][3];
                pmin = pmin.min(p);
                pmax = pmax.max(p);
            }
            // Compression gate (integration fix, approved 2026-08-13): the
            // plain min/max-p sensor cannot tell a captured shock from a
            // smooth steep expansion — on the demo nozzle it masked ~1300
            // shock-free cells at quasi-1D init, putting dissipative HLL
            // radial fluxes across the whole divergent section. A shock has
            // converging axial velocity across the window in either flow
            // direction; an expansion diverges.
            let compressing = w[g.gidx(iz + 2, ir)][1] < w[g.gidx(iz - 2, ir)][1];
            if compressing && pmin < 0.7 * pmax {
                for d in -1..=1 {
                    let j = iz + d;
                    if j >= 0 && j < nz {
                        mask[g.gidx(j, ir)] = true;
                    }
                }
            }
        }
    }
}

/// Isentropic column state [rho, u, p] (f64) at Mach `m` from the chamber
/// stagnation state. Non-dimensional: R = 1, p = rho*T, a² = gamma*T.
fn isentropic_state(m: f64, chamber: &Chamber, g: f64) -> [f64; 3] {
    let t0 = chamber.t0 as f64;
    let p0 = chamber.p0 as f64;
    let t = t0 / (1.0 + 0.5 * (g - 1.0) * m * m);
    let p = p0 * (t / t0).powf(g / (g - 1.0));
    let rho = p / t;
    let u = m * (g * t).sqrt();
    [rho, u, p]
}

/// Quasi-1D isentropic in the nozzle, ambient elsewhere, blended over 4 cells
/// at the exit plane. Open radius per column: r_w(i) = sqrt(2*sum_j
/// (1-frac[i][j])*r_j*dr_j) — r-weighted, with the square root. If the area has no
/// interior minimum (arbitrary drawn blob), fall back to ambient everywhere;
/// do not crash.
pub fn quasi1d_init(u: &mut [Cons], g: &Grid, solid: &SolidField,
                    gas: &GasModel, chamber: &Chamber, ambient: &Ambient) {
    let gm = gas.gamma;
    let gm64 = gm as f64;
    let rho_a = ambient.p / ambient.t;
    let ua = prim_to_cons([rho_a, 0.0, 0.0, ambient.p], gm);

    // Ambient everywhere first; the nozzle interior overwrites below.
    for ir in 0..g.nr {
        for iz in 0..g.nz {
            u[g.gidx(iz as isize, ir as isize)] = ua;
        }
    }

    // Open radius per column (r-weighted, docs/physics-reference.md §5) and
    // the lip: the last column containing any solid.
    let mut r_open = vec![0.0f64; g.nz];
    let mut lip: Option<usize> = None;
    for (iz, r_w) in r_open.iter_mut().enumerate() {
        let mut acc = 0.0f64; // f64 reduction
        let mut any_solid = false;
        for ir in 0..g.nr {
            let idx = g.idx(iz, ir);
            acc += (1.0 - solid.fraction[idx] as f64)
                * g.r_center(ir) as f64
                * g.dr(ir) as f64;
            any_solid |= solid.is_solid(idx);
        }
        *r_w = (2.0 * acc).sqrt();
        if any_solid {
            lip = Some(iz);
        }
    }

    // A usable nozzle needs a lip and a strictly interior area minimum with a
    // real constriction. Anything else falls back to ambient everywhere.
    let Some(lip) = lip else { return };
    if lip < 2 {
        return;
    }
    let mut i_throat = 0usize;
    for i in 0..=lip {
        if r_open[i] <= r_open[i_throat] {
            i_throat = i; // last argmin: sonic at the end of a flat throat
        }
    }
    let r_t = r_open[i_throat];
    if i_throat == 0 || i_throat == lip || r_t <= 1e-9 {
        return;
    }
    if r_t >= r_open[0].min(r_open[lip]) {
        return;
    }

    // Fill nozzle columns from the isentropic relations: subsonic upstream of
    // the throat, M = 1 analytically at it, supersonic downstream.
    let mut exit_state = [0.0f64; 3];
    for i in 0..=lip {
        let ar = (r_open[i] / r_t).powi(2);
        let m = if i == i_throat {
            1.0
        } else {
            mach_from_area_ratio(ar, gm64, i > i_throat)
        };
        let st = isentropic_state(m, chamber, gm64);
        if i == lip {
            exit_state = st;
        }
        let cons = prim_to_cons([st[0] as Real, st[1] as Real, 0.0, st[2] as Real], gm);
        // Nozzle interior = fluid cells below the first solid cell of the
        // column (below the wall). Fluid above the wall stays ambient.
        let wall_ir = (0..g.nr).find(|&ir| solid.is_solid(g.idx(i, ir)));
        for ir in 0..g.nr {
            let inside = match wall_ir {
                Some(wir) => ir < wir,
                None => (g.r_center(ir) as f64) <= r_open[i],
            };
            if inside && !solid.is_solid(g.idx(i, ir)) {
                u[g.gidx(i as isize, ir as isize)] = cons;
            }
        }
    }

    // Blend over 4 cells past the exit plane so step 1 does not launch a
    // delta-function shock. Only inside the jet radius; ambient above.
    let amb64 = [rho_a as f64, 0.0, ambient.p as f64];
    for k in 1..=4usize {
        let i = lip + k;
        if i >= g.nz {
            break;
        }
        let t = k as f64 / 5.0;
        let mut st = [0.0f64; 3];
        for c in 0..3 {
            st[c] = (1.0 - t) * exit_state[c] + t * amb64[c];
        }
        let cons = prim_to_cons([st[0] as Real, st[1] as Real, 0.0, st[2] as Real], gm);
        for ir in 0..g.nr {
            if (g.r_center(ir) as f64) <= r_open[lip] && !solid.is_solid(g.idx(i, ir)) {
                u[g.gidx(i as isize, ir as isize)] = cons;
            }
        }
    }
}

/// Area-Mach inversion, NASA b4wind, 8 fixed Newton iterations.
/// Guard |ar - 1| < 1e-6 -> return 1.0 (double root: Newton NaNs at ar = 1).
pub fn mach_from_area_ratio(ar: f64, gamma: f64, supersonic: bool) -> f64 {
    if (ar - 1.0).abs() < 1e-6 {
        return 1.0;
    }
    let pp = 2.0 / (gamma + 1.0);
    let qq = 1.0 - pp;
    let (p, q, rr) = if supersonic {
        (qq, pp, ar.powf(2.0 * qq / pp))
    } else {
        (pp, qq, ar * ar)
    };
    let e = 1.0 / q;
    let a = p.powf(e);
    let r = (rr - 1.0) / (2.0 * a);
    // Closed-form initial guess; lands in (0, 1] on both branches.
    let mut x = 1.0 / ((1.0 + r) + (r * (r + 2.0)).sqrt());
    for _ in 0..8 {
        let b = p + q * x;
        x -= (b.powf(e) - rr * x) / (b.powf(e - 1.0) - rr);
    }
    if supersonic { 1.0 / x.sqrt() } else { x.sqrt() }
}

/// What separates "conservation drift" from "the user drew a hole". Test T2
/// asserts against this, not against zero.
#[derive(Default, Debug, Clone, Copy)]
pub struct FlipLedger { pub mass: f64, pub energy: f64 }

/// Flip bookkeeping plus BFS refill (at most 8 passes) of newly opened cells
/// AT REST (u = v = 0 — an opened cell inheriting a Mach-3 neighbour's
/// velocity fires a shock into the new cavity). Remaining unreached cells get
/// ambient. Called from the top of `step()`, never inside an RK stage.
///
/// The ledger records every unit of mass/energy the edit adds or removes —
/// closures, BFS fills AND ambient fills — so that
/// `delta(total over fluid cells) == ledger` holds exactly in f64.
pub fn apply_geometry_change(u: &mut [Cons], g: &Grid, old: &SolidField,
                             new: &SolidField, gas: &GasModel,
                             ambient: &Ambient, ledger: &mut FlipLedger) {
    let gm = gas.gamma;
    let rho_a = ambient.p / ambient.t;
    let ua = prim_to_cons([rho_a, 0.0, 0.0, ambient.p], gm);
    let nz = g.nz;
    let nr = g.nr;

    // Pass 0: the flip ledger. `valid` marks fluid cells whose state is
    // trustworthy (fluid before and after the edit).
    let mut valid = vec![false; g.len()];
    let mut opened: Vec<usize> = Vec::new();
    for ir in 0..nr {
        for iz in 0..nz {
            let idx = g.idx(iz, ir);
            let gi = g.gidx(iz as isize, ir as isize);
            match (old.is_solid(idx), new.is_solid(idx)) {
                (false, true) => {
                    // fluid -> solid: its mass and energy leave the ledger.
                    let vol = g.cell_vol(iz, ir);
                    ledger.mass -= u[gi][0] as f64 * vol;
                    ledger.energy -= u[gi][3] as f64 * vol;
                    // Frozen from now on; ambient keeps primitives finite.
                    u[gi] = ua;
                }
                (true, false) => opened.push(idx),
                (false, false) => valid[idx] = true,
                (true, true) => {}
            }
        }
    }

    // BFS refill, at most 8 synchronous passes: mean rho and p over valid
    // fluid 4-neighbours, START AT REST.
    for _pass in 0..8 {
        if opened.is_empty() {
            break;
        }
        let mut still_open: Vec<usize> = Vec::new();
        let mut fills: Vec<(usize, Real, Real)> = Vec::new();
        for &idx in &opened {
            let iz = idx % nz;
            let ir = idx / nz;
            let mut rho_s = 0.0f64;
            let mut p_s = 0.0f64;
            let mut cnt = 0u32;
            let mut visit = |jz: usize, jr: usize| {
                let jdx = g.idx(jz, jr);
                if !new.is_solid(jdx) && valid[jdx] {
                    let w = cons_to_prim(u[g.gidx(jz as isize, jr as isize)], gm);
                    rho_s += w[0] as f64;
                    p_s += w[3] as f64;
                    cnt += 1;
                }
            };
            if iz > 0 { visit(iz - 1, ir); }
            if iz + 1 < nz { visit(iz + 1, ir); }
            if ir > 0 { visit(iz, ir - 1); }
            if ir + 1 < nr { visit(iz, ir + 1); }
            if cnt > 0 {
                fills.push((idx, (rho_s / cnt as f64) as Real, (p_s / cnt as f64) as Real));
            } else {
                still_open.push(idx);
            }
        }
        if fills.is_empty() {
            break;
        }
        for (idx, rho, p) in fills {
            let iz = idx % nz;
            let ir = idx / nz;
            u[g.gidx(iz as isize, ir as isize)] = prim_to_cons([rho, 0.0, 0.0, p], gm);
            valid[idx] = true;
            let vol = g.cell_vol(iz, ir);
            ledger.mass += rho as f64 * vol;
            ledger.energy += (p / (gm - 1.0)) as f64 * vol;
        }
        opened = still_open;
    }

    // Cells the BFS never reached (sealed cavities): ambient.
    for idx in opened {
        let iz = idx % nz;
        let ir = idx / nz;
        u[g.gidx(iz as isize, ir as isize)] = ua;
        let vol = g.cell_vol(iz, ir);
        ledger.mass += rho_a as f64 * vol;
        ledger.energy += ua[3] as f64 * vol;
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn gas() -> GasModel { GasModel { gamma: 1.4, r_specific_si: 287.0 } }
    fn chamber() -> Chamber { Chamber { p0: 1.0, t0: 1.0 } }
    fn ambient() -> Ambient { Ambient { p: 0.02, t: 0.6 } }
    fn numerics() -> Numerics { Numerics { quasi1d_init: false, ..Numerics::default() } }

    fn no_solid(g: &Grid) -> Vec<bool> { vec![false; g.glen()] }

    fn ambient_cons(a: &Ambient, gm: Real) -> Cons {
        prim_to_cons([a.p / a.t, 0.0, 0.0, a.p], gm)
    }

    /// Deterministic LCG so the tests need no rand crate.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) & 0xFF_FFFF) as f32 / 16_777_216.0
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 { lo + (hi - lo) * self.next() }
    }

    // ---- wall flux --------------------------------------------------------

    #[test]
    fn wall_flux_mass_and_energy_are_bit_exact_zero() {
        let mut rng = Rng(7);
        for _ in 0..100 {
            let w: Prim = [rng.range(0.01, 5.0), rng.range(-3.0, 3.0),
                           rng.range(-3.0, 3.0), rng.range(0.01, 5.0)];
            for sgn in [1.0f32, -1.0] {
                let fz = wall_flux_z(w, sgn, 1.4);
                assert!(fz[0] == 0.0 && fz[2] == 0.0 && fz[3] == 0.0, "{fz:?}");
                assert!(fz[1].is_finite() && sgn * fz[1] > 0.0);
                let fr = wall_flux_r(w, sgn, 1.4);
                assert!(fr[0] == 0.0 && fr[1] == 0.0 && fr[3] == 0.0, "{fr:?}");
                assert!(fr[2].is_finite() && sgn * fr[2] > 0.0);
            }
        }
    }

    #[test]
    fn wall_flux_star_pressure_signs() {
        let gm = 1.4f32;
        // Flow INTO the wall (fluid on low side, u > 0): ps = p + rho*un*(2un+a) > p.
        let w: Prim = [1.0, 0.5, 0.2, 1.0];
        let a = (gm * w[3] / w[0]).sqrt();
        let ps = wall_flux_z(w, 1.0, gm)[1];
        let expect = w[3] + w[0] * 0.5 * (2.0 * 0.5 + a);
        assert!((ps - expect).abs() <= 1e-6 * expect, "{ps} vs {expect}");
        assert!(ps > w[3]);
        // Receding (un < 0): ps = p - rho*|un|*a, the acoustic expansion.
        let ps = wall_flux_z(w, -1.0, gm)[1]; // fluid on high side, un = -u
        let expect = -(w[3] - w[0] * 0.5 * a);
        assert!((ps - expect).abs() <= 1e-6 * expect.abs(), "{ps} vs {expect}");
        // Strong recession floors at P_MIN_ABS instead of going negative.
        let w: Prim = [1.0, -10.0, 0.0, 1e-6];
        assert_eq!(wall_flux_z(w, 1.0, gm)[1], P_MIN_ABS);
        // wall_flux_r keys on u_r (component 2).
        let w: Prim = [1.0, 9.0, 0.5, 1.0];
        let a = (gm * w[3] / w[0]).sqrt();
        let ps = wall_flux_r(w, 1.0, gm)[2];
        let expect = w[3] + w[0] * 0.5 * (2.0 * 0.5 + a);
        assert!((ps - expect).abs() <= 1e-6 * expect, "{ps} vs {expect}");
    }

    // ---- fill_ghosts ------------------------------------------------------

    fn filled_grid(g: &Grid, f: impl Fn(usize, usize) -> Prim, gm: Real) -> Vec<Cons> {
        let mut u = vec![[0.0f32; 4]; g.glen()];
        for ir in 0..g.nr {
            for iz in 0..g.nz {
                u[g.gidx(iz as isize, ir as isize)] = prim_to_cons(f(iz, ir), gm);
            }
        }
        u
    }

    #[test]
    fn ghosts_axis_mirror_negates_radial_momentum() {
        let g = Grid::uniform(8, 6, 0.1, 0.05);
        let gm = 1.4f32;
        let mut u = filled_grid(&g, |iz, ir| {
            [1.0 + 0.1 * ir as f32, 0.3, 0.2 + 0.05 * iz as f32, 1.0 + 0.02 * iz as f32]
        }, gm);
        fill_ghosts(&mut u, &g, &no_solid(&g), &gas(), &chamber(), &ambient(), &numerics());
        for iz in 0..g.nz as isize {
            for k in 0..2isize {
                let interior = u[g.gidx(iz, k)];
                let ghost = u[g.gidx(iz, -1 - k)];
                assert_eq!(ghost[0], interior[0]);
                assert_eq!(ghost[1], interior[1]);
                assert_eq!(ghost[2], -interior[2]);
                assert_eq!(ghost[3], interior[3]);
            }
        }
    }

    #[test]
    fn ghosts_inlet_matches_stagnation_construction() {
        let g = Grid::uniform(8, 4, 0.1, 0.05);
        let gm = 1.4f32;
        let wi: Prim = [0.9, 0.3, 0.05, 0.85];
        let mut u = filled_grid(&g, |_, _| wi, gm);
        fill_ghosts(&mut u, &g, &no_solid(&g), &gas(), &chamber(), &ambient(), &numerics());
        // Reference construction in f64.
        let (gg, p0, t0) = (1.4f64, 1.0f64, 1.0f64);
        let a02 = gg * t0;
        let ai = (gg * wi[3] as f64 / wi[0] as f64).sqrt();
        let rm = wi[1] as f64 - 2.0 * ai / (gg - 1.0);
        let d = ((gg + 1.0) * a02 / (gg - 1.0) - 0.5 * (gg - 1.0) * rm * rm).max(0.0);
        let ab = (-rm + d.sqrt()) / ((gg + 1.0) / (gg - 1.0));
        let ub = (rm + 2.0 * ab / (gg - 1.0)).max(0.0);
        let pb = p0 * (ab * ab / a02).powf(gg / (gg - 1.0));
        let rb = gg * pb / (ab * ab);
        for ir in 0..g.nr as isize {
            let wg = cons_to_prim(u[g.gidx(-1, ir)], gm);
            assert!((wg[0] as f64 - rb).abs() <= 1e-5 * rb, "rho {} vs {rb}", wg[0]);
            assert!((wg[1] as f64 - ub).abs() <= 1e-5 * ub.max(1.0), "u {} vs {ub}", wg[1]);
            assert_eq!(wg[2], 0.0, "inlet tangential velocity must be zero");
            assert!((wg[3] as f64 - pb).abs() <= 1e-5 * pb, "p {} vs {pb}", wg[3]);
            assert!(wg[3] <= 1.0 + 1e-5, "inlet p above stagnation");
            assert_eq!(u[g.gidx(-2, ir)], u[g.gidx(-1, ir)]);
        }
    }

    #[test]
    fn ghosts_inlet_quiescent_chamber_reproduces_chamber_state() {
        // Interior at the chamber state (1, 0, 0, 1): the BC must hand back
        // (1, 0, 0, 1) to f32 roundoff. This is what makes T4's left state
        // exact at the inlet.
        let g = Grid::uniform(8, 2, 0.1, 0.05);
        let gm = 1.4f32;
        let mut u = filled_grid(&g, |_, _| [1.0, 0.0, 0.0, 1.0], gm);
        fill_ghosts(&mut u, &g, &no_solid(&g), &gas(), &chamber(),
                    &Ambient { p: 1.0, t: 1.0 }, &numerics());
        let wg = cons_to_prim(u[g.gidx(-1, 0)], gm);
        assert!((wg[0] - 1.0).abs() <= 1e-5, "rho {}", wg[0]);
        assert!(wg[1].abs() <= 1e-5, "u {}", wg[1]);
        assert!((wg[3] - 1.0).abs() <= 1e-5, "p {}", wg[3]);
    }

    #[test]
    fn ghosts_inlet_no_nan_under_reverse_flow() {
        let g = Grid::uniform(8, 2, 0.1, 0.05);
        let gm = 1.4f32;
        for i in 0..50 {
            let ui = -5.0 + 0.1 * i as f32; // strong reverse flow sweep
            let mut u = filled_grid(&g, |_, _| [0.8, ui, 0.1, 0.7], gm);
            fill_ghosts(&mut u, &g, &no_solid(&g), &gas(), &chamber(), &ambient(), &numerics());
            let wg = cons_to_prim(u[g.gidx(-1, 0)], gm);
            assert!(wg.iter().all(|v| v.is_finite()), "u_i = {ui}: {wg:?}");
            assert!(wg[0] > 0.0 && wg[3] > 0.0, "u_i = {ui}: {wg:?}");
            assert!(wg[1] >= 0.0, "u_i = {ui}: inlet velocity clamped at zero");
        }
    }

    #[test]
    fn ghosts_outflow_supersonic_copies_interior_exactly() {
        let g = Grid::uniform(8, 4, 0.1, 0.05);
        let gm = 1.4f32;
        let wi: Prim = [0.5, 2.5, 0.3, 0.4]; // a ~ 1.06, u > a
        let mut u = filled_grid(&g, |_, _| wi, gm);
        let interior = u[g.gidx(g.nz as isize - 1, 1)];
        fill_ghosts(&mut u, &g, &no_solid(&g), &gas(), &chamber(), &ambient(), &numerics());
        assert_eq!(u[g.gidx(g.nz as isize, 1)], interior);
        assert_eq!(u[g.gidx(g.nz as isize + 1, 1)], interior);
    }

    #[test]
    fn ghosts_outflow_subsonic_imposes_ambient_pressure() {
        let g = Grid::uniform(8, 4, 0.1, 0.05);
        let gm = 1.4f32;
        let amb = Ambient { p: 0.7, t: 0.9 };
        let wi: Prim = [1.0, 0.3, 0.1, 0.9];
        let mut u = filled_grid(&g, |_, _| wi, gm);
        fill_ghosts(&mut u, &g, &no_solid(&g), &gas(), &chamber(), &amb, &numerics());
        let wg = cons_to_prim(u[g.gidx(g.nz as isize, 1)], gm);
        let rb = wi[0] as f64 * (0.7f64 / wi[3] as f64).powf(1.0 / 1.4);
        let ai = (1.4 * wi[3] as f64 / wi[0] as f64).sqrt();
        let ab = (1.4 * 0.7 / rb).sqrt();
        let ub = wi[1] as f64 + 2.0 * (ai - ab) / 0.4;
        assert!((wg[3] as f64 - 0.7).abs() <= 1e-5, "p_b = {}", wg[3]);
        assert!((wg[0] as f64 - rb).abs() <= 1e-5 * rb, "rho_b = {}", wg[0]);
        assert!((wg[1] as f64 - ub).abs() <= 1e-5 * ub.abs().max(1.0), "u_b = {}", wg[1]);
        assert!((wg[2] - wi[2]).abs() <= 1e-6, "v_b extrapolates v_i");
    }

    #[test]
    fn ghosts_outflow_reversed_flow_uses_ambient_reservoir() {
        // The A2 fix: any reversed flow at the outflow plane — subsonic OR
        // supersonic — gets the ambient reservoir, exactly like the radial
        // far field. Before it, u = -0.30 here produced a ghost jetting
        // OUTWARD at M 2.5, and u = -2.0 (|u| > a) copied the interior
        // bit-exactly, imposing nothing on four incoming characteristics.
        let g = Grid::uniform(8, 4, 0.1, 0.05);
        let gm = 1.4f32;
        let amb = ambient();
        let expect = ambient_cons(&amb, gm);
        // Subsonic reversal (a ~ 1.06 for this state).
        let mut u = filled_grid(&g, |_, _| [0.8, -0.30, 0.4, 0.65], gm);
        fill_ghosts(&mut u, &g, &no_solid(&g), &gas(), &chamber(), &amb, &numerics());
        assert_eq!(u[g.gidx(g.nz as isize, 1)], expect);
        assert_eq!(u[g.gidx(g.nz as isize + 1, 1)], expect);
        // Supersonic reversal: |u| >= a must NOT read as supersonic outflow.
        let mut u = filled_grid(&g, |_, _| [1.0, -2.0, 0.1, 0.06], gm); // a ~ 0.29
        fill_ghosts(&mut u, &g, &no_solid(&g), &gas(), &chamber(), &amb, &numerics());
        assert_eq!(u[g.gidx(g.nz as isize, 1)], expect);
        // Outgoing flow at exactly u = 0 still takes the subsonic outflow
        // branch (imposes p_a), not the reservoir.
        let wi: Prim = [1.0, 0.0, 0.2, 0.9];
        let mut u = filled_grid(&g, |_, _| wi, gm);
        fill_ghosts(&mut u, &g, &no_solid(&g), &gas(), &chamber(), &amb, &numerics());
        let wg = cons_to_prim(u[g.gidx(g.nz as isize, 1)], gm);
        assert!((wg[3] - amb.p).abs() <= 1e-6, "p_b = {}", wg[3]);
        assert!((wg[2] - wi[2]).abs() <= 1e-6, "v_b extrapolates v_i at u = 0");
    }

    #[test]
    fn ghosts_farfield_inflow_uses_ambient_reservoir() {
        let g = Grid::uniform(8, 4, 0.1, 0.05);
        let gm = 1.4f32;
        let amb = ambient();
        // v < 0: entering. Ghost is the ambient reservoir at rest.
        let mut u = filled_grid(&g, |_, _| [0.5, 0.2, -0.1, 0.3], gm);
        fill_ghosts(&mut u, &g, &no_solid(&g), &gas(), &chamber(), &amb, &numerics());
        let expect = ambient_cons(&amb, gm);
        assert_eq!(u[g.gidx(3, g.nr as isize)], expect);
        assert_eq!(u[g.gidx(3, g.nr as isize + 1)], expect);
        // 0 <= v < a: subsonic outflow, impose p_a with the radial normal.
        let wi: Prim = [0.5, 0.2, 0.1, 0.3];
        let mut u = filled_grid(&g, |_, _| wi, gm);
        fill_ghosts(&mut u, &g, &no_solid(&g), &gas(), &chamber(), &amb, &numerics());
        let wg = cons_to_prim(u[g.gidx(3, g.nr as isize)], gm);
        assert!((wg[3] - amb.p).abs() <= 1e-6, "p_b = {}", wg[3]);
        assert!((wg[1] - wi[1]).abs() <= 1e-6, "u_z is tangential here");
        // v >= a: supersonic outflow copies the interior.
        let mut u = filled_grid(&g, |_, _| [0.5, 0.2, 2.0, 0.3], gm);
        let interior = u[g.gidx(3, g.nr as isize - 1)];
        fill_ghosts(&mut u, &g, &no_solid(&g), &gas(), &chamber(), &amb, &numerics());
        assert_eq!(u[g.gidx(3, g.nr as isize)], interior);
    }

    // ---- sponge -----------------------------------------------------------

    #[test]
    fn sponge_integrated_strength_is_four_and_interior_untouched() {
        // Sum of sigma*dr/a over the 24 rows must be ~4: e^-4 = 1.8% one-way
        // transmission, the docs/physics-reference.md §4 target. The per-step
        // form this replaces accumulates only ~0.9.
        let g = Grid::uniform(4, 30, 0.1, 0.05);
        let gm = 1.4f32;
        let amb = ambient();
        let ua = ambient_cons(&amb, gm);
        let cells = 24usize;
        let mut u = vec![[0.0f32; 4]; g.glen()];
        for ir in 0..g.nr {
            for iz in 0..g.nz {
                let c = &mut u[g.gidx(iz as isize, ir as isize)];
                *c = ua;
                c[0] += 1.0; // unit perturbation to watch decay
            }
        }
        let before = u.clone();
        let dt = 1e-3f32;
        apply_sponge(&mut u, &g, dt, &amb, &gas(), cells);
        let a_amb = (gm as f64 * amb.t as f64).sqrt();
        let mut total = 0.0f64;
        let mut prev = -1.0f64;
        for ir in 0..g.nr {
            let c = u[g.gidx(1, ir as isize)];
            let decrement = before[g.gidx(1, ir as isize)][0] as f64 - c[0] as f64;
            if ir < g.nr - cells {
                assert_eq!(decrement, 0.0, "row {ir} outside the sponge was touched");
            } else {
                assert!(decrement > prev, "sigma must grow with depth");
                prev = decrement;
                total += decrement / dt as f64 * (g.dr(ir) as f64 / a_amb);
            }
        }
        assert!((total - 4.0).abs() < 0.1, "integrated sponge strength = {total}");
    }

    #[test]
    fn sponge_zero_cells_is_a_no_op() {
        let g = Grid::uniform(4, 8, 0.1, 0.05);
        let gm = 1.4f32;
        let mut u = filled_grid(&g, |iz, ir| [1.0 + 0.1 * ir as f32, 0.5, 0.2, 1.0 + 0.01 * iz as f32], gm);
        let before = u.clone();
        apply_sponge(&mut u, &g, 0.01, &ambient(), &gas(), 0);
        assert_eq!(u, before);
    }

    // ---- carbuncle sensor -------------------------------------------------

    #[test]
    fn carbuncle_mask_fires_three_cells_around_a_shock_only() {
        let g = Grid::uniform(30, 3, 0.1, 0.05);
        let mut w = vec![[1.0f32, 2.0, 0.0, 1.0]; g.glen()];
        // Shock-like jump at iz = 10, row 1 only: pressure 1.0 -> 0.5 with
        // DECELERATING axial velocity (the compression gate requires it; a
        // pressure jump with diverging velocity is an expansion).
        for iz in 10..(g.nz as isize + NG as isize) {
            w[g.gidx(iz, 1)][3] = 0.5;
            w[g.gidx(iz, 1)][1] = 1.0;
        }
        let mut mask = vec![false; g.glen()];
        carbuncle_mask(&w, &g, &mut mask);
        for iz in 0..g.nz as isize {
            let expect = (7..=12).contains(&iz);
            assert_eq!(mask[g.gidx(iz, 1)], expect, "row 1, iz = {iz}");
            assert!(!mask[g.gidx(iz, 0)], "row 0 must stay unmasked");
            assert!(!mask[g.gidx(iz, 2)], "row 2 must stay unmasked");
        }
    }

    #[test]
    fn carbuncle_mask_quiet_on_steep_smooth_expansion() {
        let g = Grid::uniform(30, 3, 0.1, 0.05);
        let mut w = vec![[1.0f32, 0.5, 0.0, 1.0]; g.glen()];
        // Nozzle-like expansion: over 5 cells p falls by 2x (Omega = 0.5
        // < 0.7 — the old sensor fired) while u RISES. Must stay unmasked.
        for ir in -2..(g.nr as isize + 2) {
            for iz in -2..(g.nz as isize + 2) {
                let f = 0.87f32.powi(iz as i32 + 2);
                w[g.gidx(iz, ir)][3] = f;
                w[g.gidx(iz, ir)][1] = 0.5 + 0.1 * (iz + 2) as f32;
            }
        }
        let mut mask = vec![true; g.glen()];
        carbuncle_mask(&w, &g, &mut mask);
        assert!(mask.iter().all(|&m| !m), "sensor fired on a smooth expansion");
    }

    #[test]
    fn carbuncle_mask_quiet_on_smooth_pressure() {
        let g = Grid::uniform(20, 3, 0.1, 0.05);
        let mut w = vec![[1.0f32, 0.0, 0.0, 1.0]; g.glen()];
        // 5% ripple: Omega ~ 0.9 > 0.7 everywhere.
        for ir in -2..(g.nr as isize + 2) {
            for iz in -2..(g.nz as isize + 2) {
                w[g.gidx(iz, ir)][3] = 1.0 + 0.05 * ((iz * 3 + ir) % 2) as f32;
            }
        }
        let mut mask = vec![true; g.glen()]; // must be cleared by the call
        carbuncle_mask(&w, &g, &mut mask);
        assert!(mask.iter().all(|&m| !m));
    }

    // ---- area-Mach inversion ----------------------------------------------

    fn area_ratio_of(m: f64, g: f64) -> f64 {
        (1.0 / m) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * m * m))
            .powf((g + 1.0) / (2.0 * (g - 1.0)))
    }

    #[test]
    fn mach_area_round_trip_to_1e11() {
        // 1e-11, not 3e-13: the tighter figure holds only at gamma = 1.4; at
        // gamma = 1.2 near sonic it reaches 2e-12 (physics-reference §5).
        for g in [1.2f64, 1.24, 1.4] {
            for &m in &[0.005f64, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99,
                        1.01, 1.1, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 15.0] {
                let ar = area_ratio_of(m, g);
                let back = mach_from_area_ratio(ar, g, m > 1.0);
                assert!((back - m).abs() <= 1e-11 * m.max(1.0),
                        "gamma {g}, M {m}: round trip {back}");
            }
        }
    }

    #[test]
    fn mach_area_sonic_guard_returns_one() {
        for g in [1.2f64, 1.24, 1.4] {
            for sup in [false, true] {
                assert_eq!(mach_from_area_ratio(1.0, g, sup), 1.0);
                assert_eq!(mach_from_area_ratio(1.0 + 5e-7, g, sup), 1.0);
                assert_eq!(mach_from_area_ratio(1.0 - 5e-7, g, sup), 1.0);
            }
        }
    }

    // ---- quasi-1D init ----------------------------------------------------

    /// Converging-diverging test nozzle: wall row per column, solid above.
    /// r_open(iz) = wall*dr exactly, so area ratios are exact by construction.
    fn cd_nozzle(g: &Grid, wall: impl Fn(usize) -> Option<usize>) -> SolidField {
        let mut s = SolidField::empty(g.clone());
        for iz in 0..g.nz {
            if let Some(w) = wall(iz) {
                for ir in w..g.nr {
                    s.fraction[g.idx(iz, ir)] = 1.0;
                }
            }
        }
        s
    }

    fn cd_wall(iz: usize) -> Option<usize> {
        match iz {
            0..=10 => Some(16),
            11..=20 => Some(16 - (iz - 10) * 8 / 10), // converge 16 -> 8
            21..=22 => Some(8),                       // flat throat
            23..=40 => Some(8 + (iz - 22) * 6 / 18),  // diverge 8 -> 14, lip at 40
            _ => None,
        }
    }

    #[test]
    fn quasi1d_fills_a_cd_nozzle_isentropically() {
        let g = Grid::uniform(60, 24, 0.1, 0.05);
        let gm = 1.4f32;
        let amb = ambient();
        let solid = cd_nozzle(&g, cd_wall);
        let mut u = vec![[0.0f32; 4]; g.glen()];
        quasi1d_init(&mut u, &g, &solid, &gas(), &chamber(), &amb);

        let prim_at = |iz: usize, ir: usize| cons_to_prim(u[g.gidx(iz as isize, ir as isize)], gm);
        // Throat (last argmin = iz 22): sonic. p/p0 = (2/(g+1))^(g/(g-1)).
        let w = prim_at(22, 0);
        let p_crit = (2.0f64 / 2.4).powf(3.5);
        assert!((w[3] as f64 - p_crit).abs() < 1e-5, "throat p = {} vs {p_crit}", w[3]);
        let a = (gm * w[3] / w[0]).sqrt();
        assert!((w[1] / a - 1.0).abs() < 1e-4, "throat M = {}", w[1] / a);
        // Chamber (area ratio 4): subsonic, near stagnation, flowing downstream.
        let w = prim_at(5, 0);
        assert!(w[3] > 0.95 && w[1] > 0.0 && w[1] < 0.3, "chamber state {w:?}");
        // Lip column (area ratio (14/8)^2): supersonic.
        let w = prim_at(40, 0);
        let a = (gm * w[3] / w[0]).sqrt();
        assert!(w[1] / a > 2.0 && w[1] / a < 3.0, "exit M = {}", w[1] / a);
        // Above the wall inside the nozzle span: ambient.
        let w = prim_at(45, 20);
        assert!((w[3] - amb.p).abs() < 1e-6, "outside-nozzle p = {}", w[3]);
        // Past the 4-cell exit blend: ambient exactly.
        assert_eq!(u[g.gidx(50, 0)], ambient_cons(&amb, gm));
        // Blend cells sit between exit and ambient pressure, monotone.
        let p_exit = prim_at(40, 0)[3];
        let mut prev = p_exit;
        for k in 1..=4usize {
            let p = prim_at(40 + k, 0)[3];
            assert!(p <= prev.max(amb.p) + 1e-6 && p >= prev.min(amb.p) - 1e-6,
                    "blend cell {k} p = {p}");
            prev = p;
        }
        // Everything finite and positive.
        for ir in 0..g.nr {
            for iz in 0..g.nz {
                let w = prim_at(iz, ir);
                assert!(w[0] > 0.0 && w[3] > 0.0 && w.iter().all(|v| v.is_finite()),
                        "({iz}, {ir}): {w:?}");
            }
        }
    }

    #[test]
    fn quasi1d_falls_back_to_ambient_without_interior_throat() {
        let g = Grid::uniform(60, 24, 0.1, 0.05);
        let gm = 1.4f32;
        let amb = ambient();
        let ua = ambient_cons(&amb, gm);
        // Converging-only: area minimum at the lip, not interior.
        let solid = cd_nozzle(&g, |iz| if iz <= 30 { Some(16 - iz / 5) } else { None });
        let mut u = vec![[0.0f32; 4]; g.glen()];
        quasi1d_init(&mut u, &g, &solid, &gas(), &chamber(), &amb);
        for ir in 0..g.nr {
            for iz in 0..g.nz {
                if !solid.is_solid(g.idx(iz, ir)) {
                    assert_eq!(u[g.gidx(iz as isize, ir as isize)], ua, "({iz}, {ir})");
                }
            }
        }
        // No geometry at all: also ambient, no crash.
        let empty = SolidField::empty(g.clone());
        let mut u = vec![[0.0f32; 4]; g.glen()];
        quasi1d_init(&mut u, &g, &empty, &gas(), &chamber(), &amb);
        assert_eq!(u[g.gidx(30, 10)], ua);
    }

    // ---- geometry change --------------------------------------------------

    /// f64 total mass/energy over the fluid cells of `solid`.
    fn totals(u: &[Cons], g: &Grid, solid: &SolidField) -> (f64, f64) {
        let mut mass = 0.0f64;
        let mut energy = 0.0f64;
        for ir in 0..g.nr {
            for iz in 0..g.nz {
                if solid.is_solid(g.idx(iz, ir)) {
                    continue;
                }
                let c = u[g.gidx(iz as isize, ir as isize)];
                let vol = g.cell_vol(iz, ir);
                mass += c[0] as f64 * vol;
                energy += c[3] as f64 * vol;
            }
        }
        (mass, energy)
    }

    fn block_solid(g: &Grid, iz0: usize, iz1: usize, ir0: usize, ir1: usize) -> SolidField {
        let mut s = SolidField::empty(g.clone());
        for ir in ir0..=ir1 {
            for iz in iz0..=iz1 {
                s.fraction[g.idx(iz, ir)] = 1.0;
            }
        }
        s
    }

    #[test]
    fn geometry_close_books_removed_mass_on_the_ledger() {
        let g = Grid::uniform(10, 6, 0.1, 0.05);
        let gm = 1.4f32;
        let old = SolidField::empty(g.clone());
        let new = block_solid(&g, 4, 6, 2, 3);
        let mut u = filled_grid(&g, |iz, ir| {
            [1.0 + 0.01 * (ir * 10 + iz) as f32, 0.2, -0.1, 1.0 + 0.005 * iz as f32]
        }, gm);
        let (m0, e0) = totals(&u, &g, &old);
        let mut ledger = FlipLedger::default();
        apply_geometry_change(&mut u, &g, &old, &new, &gas(), &ambient(), &mut ledger);
        let (m1, e1) = totals(&u, &g, &new);
        assert!(ledger.mass < 0.0, "closing cells must remove mass");
        assert!(((m1 - m0) - ledger.mass).abs() <= 1e-12 * m0, "T2 invariant (mass)");
        assert!(((e1 - e0) - ledger.energy).abs() <= 1e-12 * e0, "T2 invariant (energy)");
    }

    #[test]
    fn geometry_open_fills_at_rest_from_neighbour_means() {
        let g = Grid::uniform(10, 6, 0.1, 0.05);
        let gm = 1.4f32;
        let old = block_solid(&g, 4, 6, 2, 3);
        let new = SolidField::empty(g.clone());
        let amb = ambient();
        let mut u = filled_grid(&g, |iz, ir| {
            [1.0 + 0.01 * (ir * 10 + iz) as f32, 0.4, -0.2, 1.0 + 0.005 * iz as f32]
        }, gm);
        // Solid cells hold ambient, as they would in a live run.
        for ir in 2..=3usize {
            for iz in 4..=6usize {
                u[g.gidx(iz as isize, ir as isize)] = ambient_cons(&amb, gm);
            }
        }
        let (m0, e0) = totals(&u, &g, &old);
        let mut ledger = FlipLedger::default();
        apply_geometry_change(&mut u, &g, &old, &new, &gas(), &amb, &mut ledger);
        let (m1, e1) = totals(&u, &g, &new);
        assert!(ledger.mass > 0.0, "opening cells must add mass");
        assert!(((m1 - m0) - ledger.mass).abs() <= 1e-12 * m0.max(1.0), "T2 invariant (mass)");
        assert!(((e1 - e0) - ledger.energy).abs() <= 1e-12 * e0.max(1.0), "T2 invariant (energy)");
        // Every opened cell starts AT REST with the mean of its valid fluid
        // neighbours; corner cell (4, 2) has exactly (3, 2) and (4, 1).
        for ir in 2..=3usize {
            for iz in 4..=6usize {
                let c = u[g.gidx(iz as isize, ir as isize)];
                assert_eq!(c[1], 0.0, "({iz}, {ir}) momentum z");
                assert_eq!(c[2], 0.0, "({iz}, {ir}) momentum r");
                assert!(c[0] > 0.0 && c[3] > 0.0);
            }
        }
        let wa = cons_to_prim(u[g.gidx(3, 2)], gm);
        let wb = cons_to_prim(u[g.gidx(4, 1)], gm);
        let wc = cons_to_prim(u[g.gidx(4, 2)], gm);
        assert!((wc[0] - 0.5 * (wa[0] + wb[0])).abs() <= 1e-6, "mean rho");
        assert!((wc[3] - 0.5 * (wa[3] + wb[3])).abs() <= 1e-6, "mean p");
    }

    #[test]
    fn geometry_open_sealed_cavity_gets_ambient() {
        let g = Grid::uniform(10, 8, 0.1, 0.05);
        let gm = 1.4f32;
        let amb = ambient();
        let old = block_solid(&g, 3, 5, 2, 4); // 3x3 solid block
        let mut new = block_solid(&g, 3, 5, 2, 4);
        new.fraction[g.idx(4, 3)] = 0.0; // open the centre; ring stays solid
        let mut u = filled_grid(&g, |_, _| [0.9, 0.5, 0.1, 0.8], gm);
        let mut ledger = FlipLedger::default();
        apply_geometry_change(&mut u, &g, &old, &new, &gas(), &amb, &mut ledger);
        assert_eq!(u[g.gidx(4, 3)], ambient_cons(&amb, gm));
        let vol = g.cell_vol(4, 3);
        let rho_a = (amb.p / amb.t) as f64;
        assert!((ledger.mass - rho_a * vol).abs() <= 1e-12, "cavity mass on the ledger");
    }
}
