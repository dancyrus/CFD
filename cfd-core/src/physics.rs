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

/// True when every column has at most one solid run — the star-convex-in-r
/// assumption `column_reflect_fill` depends on. `EulerSolver::step` REFUSES
/// `WallMode::ColumnReflect` when this fails (work order A2): the fill finds
/// the first solid cell from the axis and overwrites everything above it, so
/// on a column with a second fluid run it destroys live fluid — measured on a
/// 9-disk array, 2346 of 21420 live fluid cells overwritten and the report
/// off by 2.6x, silently. On the nozzle: 0 of 15977 cells, which is why no
/// ladder rung ever saw it.
pub fn column_reflect_supported(g: &Grid, solid: &SolidField) -> bool {
    for iz in 0..g.nz {
        let mut runs = 0usize;
        let mut prev_solid = false;
        for ir in 0..g.nr {
            let s = solid.is_solid(g.idx(iz, ir));
            if s && !prev_solid {
                runs += 1;
                if runs > 1 {
                    return false;
                }
            }
            prev_solid = s;
        }
    }
    true
}

/// WallMode::ColumnReflect (abort-ladder rung 3, also used as a measurement of
/// the staircase wall's own contribution to the wall layer): per column, every
/// solid cell is overwritten with the mirror image of the fluid below the bore
/// (u_r negated), so the sweeps can run the REGULAR flux solver at wall faces
/// and the bore face becomes a reflecting condition. Only star-convex-in-r
/// geometry survives — `column_reflect_supported` is the gate and
/// `EulerSolver::step` refuses the mode when it fails, so this fill never
/// sees a column with a second solid run. Solid cells stay frozen (accumulate
/// skips them) and are refilled here before every stage.
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

/// Mach of a stream expanded isentropically from the chamber stagnation state
/// down to static pressure `p`. `None` when the expansion is undefined
/// (p <= 0, or p at/above stagnation).
fn isentropic_exit_mach(p: f64, chamber: &Chamber, g: f64) -> Option<f64> {
    let p0 = chamber.p0 as f64;
    if !(p > 0.0) || p >= p0 {
        return None;
    }
    Some((2.0 / (g - 1.0) * ((p0 / p).powf((g - 1.0) / g) - 1.0)).sqrt())
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

/// The geometric description of a nozzle the quasi-1D machinery needs: the
/// r-weighted open radius of every column, the lip, and the throat column.
#[derive(Debug, Clone)]
pub struct NozzleProfile {
    /// Open radius per column, `g.nz` long: `sqrt(2*sum_j (1-frac)*r_j*dr_j)`
    /// (docs/physics-reference.md §5). Defined for every column, nozzle or not.
    pub r_open: Vec<f64>,
    /// Last column containing any solid cell.
    pub lip: usize,
    /// Throat column: the LAST argmin of `r_open` over `0..=lip`, so a flat
    /// throat goes sonic at its downstream end.
    pub i_throat: usize,
}

/// r-weighted open radius per column and the lip. Defined for any geometry;
/// `nozzle_profile` is what decides whether the numbers mean anything.
fn open_radius_profile(g: &Grid, solid: &SolidField) -> (Vec<f64>, Option<usize>) {
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
    (r_open, lip)
}

/// Number of maximal runs of consecutive fluid cells in column `iz`.
fn fluid_runs_in_column(g: &Grid, solid: &SolidField, iz: usize) -> usize {
    let mut runs = 0usize;
    let mut prev_solid = true;
    for ir in 0..g.nr {
        let s = solid.is_solid(g.idx(iz, ir));
        if prev_solid && !s {
            runs += 1;
        }
        prev_solid = s;
    }
    runs
}

/// **The** nozzle predicate (work order A2). `quasi1d_init` gates on it, and
/// work order A3 needs the identical test — call this, never restate it.
///
/// `Some(profile)` when all of these hold:
///
///   1. A lip exists — some column contains solid — and it is at least the
///      third column, so there is an upstream chamber to expand from.
///   2. **Every column from the inlet to the lip has exactly one fluid run.**
///      A column through a disk array or a mid-duct baffle has two or more; a
///      fully blocked column has none. This is the assumption a per-column
///      bore scan makes silently, so it is checked instead of assumed.
///   3. **The open-radius profile converges then diverges**: non-increasing
///      up to the throat, non-decreasing from it to the lip.
///   4. The throat is a real constriction — nonzero, strictly interior, and
///      strictly smaller than both ends.
///
/// The gate this replaces asked only for a lip, an interior argmin and (4), so
/// a blockage anywhere satisfied it. Measured with the old gate: a 3x3 array
/// of disconnected disks was seeded as a "nozzle" over 73.1% of its fluid
/// cells at up to M 2.07, and a baffled duct over 98.4% of its cells at up to
/// M 3.13 and 0.996 p0 against an ambient of 0.0203 p0 — a 50x overpressure
/// blast wave launched at t = 0 into a case with no nozzle in it, on by
/// default.
pub fn nozzle_profile(g: &Grid, solid: &SolidField) -> Option<NozzleProfile> {
    let (r_open, lip) = open_radius_profile(g, solid);
    let lip = lip?;
    if lip < 2 {
        return None;
    }
    // (2) exactly one fluid run per column, inlet through lip. Columns past
    // the lip are plume, not nozzle; the fill never writes inside solid there.
    if (0..=lip).any(|iz| fluid_runs_in_column(g, solid, iz) != 1) {
        return None;
    }
    let mut i_throat = 0usize;
    for i in 0..=lip {
        if r_open[i] <= r_open[i_throat] {
            i_throat = i; // last argmin: sonic at the end of a flat throat
        }
    }
    // (4) a real constriction, strictly interior.
    let r_t = r_open[i_throat];
    if i_throat == 0 || i_throat == lip || r_t <= 1e-9 {
        return None;
    }
    if r_t >= r_open[0].min(r_open[lip]) {
        return None;
    }
    // (3) converge then diverge, no tolerance: the exact-area rasterizer
    // gives a monotone open radius on each side of a real CD wall, and
    // admitting "nearly monotone" is what would let a baffled duct back in.
    if (0..i_throat).any(|i| r_open[i + 1] > r_open[i]) {
        return None;
    }
    if (i_throat..lip).any(|i| r_open[i + 1] < r_open[i]) {
        return None;
    }
    Some(NozzleProfile { r_open, lip, i_throat })
}

/// Quasi-1D isentropic init in the nozzle, ambient elsewhere, blended over 4
/// cells at the exit plane — gated on `nozzle_profile`. When the gate fails
/// (arbitrary drawn geometry: disk arrays, baffles, blobs), a generic initial
/// condition instead: uniform isentropic freestream if the chamber-to-ambient
/// expansion is supersonic, ambient at rest otherwise. Either path logs once —
/// a wrong init here is invisible after a few thousand steps and the log line
/// is the only record of which path ran. Never crashes on any geometry.
pub fn quasi1d_init(u: &mut [Cons], g: &Grid, solid: &SolidField,
                    gas: &GasModel, chamber: &Chamber, ambient: &Ambient) {
    let gm = gas.gamma;
    let gm64 = gm as f64;
    let rho_a = ambient.p / ambient.t;
    let ua = prim_to_cons([rho_a, 0.0, 0.0, ambient.p], gm);

    // Ambient everywhere first; the chosen init overwrites fluid cells below.
    for ir in 0..g.nr {
        for iz in 0..g.nz {
            u[g.gidx(iz as isize, ir as isize)] = ua;
        }
    }

    let Some(profile) = nozzle_profile(g, solid) else {
        // Generic fallback. Uniform freestream when the inlet feeds a
        // supersonic stream — meaning the chamber-to-ambient expansion is
        // supersonic AND the ambient actually lies on the chamber's isentrope
        // (the external-flow rig: chamber = the stream's stagnation state,
        // ambient = its static state, and the freestream at that Mach is the
        // near-solution start). A rocket chamber over a low ambient also has
        // a supersonic pressure ratio, but its ambient is NOT the chamber's
        // stream — uniform M 3 flow through arbitrary drawn geometry would be
        // the same blast-wave mistake the gate exists to prevent, so that
        // case gets ambient at rest.
        let m = isentropic_exit_mach(ambient.p as f64, chamber, gm64);
        let on_isentrope = |m: f64| {
            let t_isen = chamber.t0 as f64 / (1.0 + 0.5 * (gm64 - 1.0) * m * m);
            ((ambient.t as f64) - t_isen).abs() <= 1e-3 * t_isen
        };
        match m {
            Some(m) if m >= 1.0 && on_isentrope(m) => {
                let st = isentropic_state(m, chamber, gm64);
                let cons = prim_to_cons([st[0] as Real, st[1] as Real, 0.0,
                                         st[2] as Real], gm);
                for ir in 0..g.nr {
                    for iz in 0..g.nz {
                        if !solid.is_solid(g.idx(iz, ir)) {
                            u[g.gidx(iz as isize, ir as isize)] = cons;
                        }
                    }
                }
                eprintln!(
                    "cfd-core: quasi-1D init: geometry is not a nozzle \
                     (gate: one fluid run per column + converge-then-diverge \
                     open radius); initialized uniform freestream at M {m:.3}"
                );
            }
            _ => {
                eprintln!(
                    "cfd-core: quasi-1D init: geometry is not a nozzle \
                     (gate: one fluid run per column + converge-then-diverge \
                     open radius); initialized ambient at rest"
                );
            }
        }
        return;
    };
    eprintln!(
        "cfd-core: quasi-1D init: nozzle gate passed (lip column {}, throat \
         column {}); initialized quasi-1D isentropic",
        profile.lip, profile.i_throat
    );
    let NozzleProfile { r_open, lip, i_throat } = profile;
    let r_t = r_open[i_throat];

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
///
/// Momentum is booked as well as mass and energy (work order A2). Closing a
/// cell deletes its momentum outright, and every refill starts at rest —
/// without momentum lines a large edit silently injects or removes momentum
/// with no audit trail, and the invariant that catches it
/// (`delta(total) == ledger`) was not even defined for momentum.
#[derive(Default, Debug, Clone, Copy)]
pub struct FlipLedger {
    pub mass: f64,
    pub energy: f64,
    pub momentum_z: f64,
    pub momentum_r: f64,
}

/// Flip bookkeeping plus BFS refill of newly opened cells AT REST (u = v = 0 —
/// an opened cell inheriting a Mach-3 neighbour's velocity fires a shock into
/// the new cavity). Called from the top of `step()`, never inside an RK stage.
///
/// The BFS runs **to completion** (work order A2). It was capped at 8 passes,
/// which covers nudging a wall and nothing else — measured fractions of an
/// erased solid block left unreached were 0% at 6x6 and 16x16, 48% at 40x40,
/// 80% at 80x80, and erasing an obstacle is exactly what a sandbox user does.
/// The loop terminates on its own: every pass fills at least one cell or
/// breaks.
///
/// Cells the flood fill can never reach (sealed cavities with no valid fluid
/// neighbour on any pass) take the (rho, p) of the NEAREST valid fluid cell by
/// grid distance, at rest, found by a multi-source BFS that ignores solidity —
/// not ambient: a sealed void inside a hot body filled with cold far-field
/// ambient is a spurious pressure discontinuity the solver then has to
/// swallow. Ambient remains only for the degenerate edit that leaves no valid
/// fluid cell anywhere.
///
/// The ledger records every unit of mass/momentum/energy the edit adds or
/// removes — closures and every kind of fill — so that
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
                    // fluid -> solid: mass, momentum and energy leave the ledger.
                    let vol = g.cell_vol(iz, ir);
                    ledger.mass -= u[gi][0] as f64 * vol;
                    ledger.momentum_z -= u[gi][1] as f64 * vol;
                    ledger.momentum_r -= u[gi][2] as f64 * vol;
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

    // BFS refill, synchronous passes TO COMPLETION: mean rho and p over valid
    // fluid 4-neighbours, START AT REST. Terminates on its own — every pass
    // fills at least one cell or breaks on `fills.is_empty()`.
    loop {
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

    // Cells the flood fill never reached (sealed cavities): extrapolate from
    // the NEAREST valid fluid cell by grid distance, at rest — a multi-source
    // BFS from every valid cell, through solid and unreached cells alike.
    // Ambient only if the edit left no valid fluid cell anywhere.
    if !opened.is_empty() {
        let mut nearest: Vec<usize> = vec![usize::MAX; g.len()];
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        for idx in 0..g.len() {
            if valid[idx] {
                nearest[idx] = idx;
                queue.push_back(idx);
            }
        }
        while let Some(idx) = queue.pop_front() {
            let iz = idx % nz;
            let ir = idx / nz;
            let mut push = |jz: usize, jr: usize| {
                let jdx = g.idx(jz, jr);
                if nearest[jdx] == usize::MAX {
                    nearest[jdx] = nearest[idx];
                    queue.push_back(jdx);
                }
            };
            if iz > 0 { push(iz - 1, ir); }
            if iz + 1 < nz { push(iz + 1, ir); }
            if ir > 0 { push(iz, ir - 1); }
            if ir + 1 < nr { push(iz, ir + 1); }
        }
        for idx in opened {
            let iz = idx % nz;
            let ir = idx / nz;
            let fill = match nearest[idx] {
                usize::MAX => ua,
                src => {
                    let sz = src % nz;
                    let sr = src / nz;
                    let w = cons_to_prim(u[g.gidx(sz as isize, sr as isize)], gm);
                    prim_to_cons([w[0], 0.0, 0.0, w[3]], gm)
                }
            };
            u[g.gidx(iz as isize, ir as isize)] = fill;
            let vol = g.cell_vol(iz, ir);
            ledger.mass += fill[0] as f64 * vol;
            ledger.energy += fill[3] as f64 * vol;
        }
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

    #[test]
    fn column_reflect_gate_counts_solid_runs() {
        let g = Grid::uniform(20, 10, 0.1, 0.05);
        // Nozzle-like: one solid run at the top of some columns.
        let mut noz = SolidField::empty(g.clone());
        for iz in 5..15usize {
            for ir in 6..10usize {
                noz.fraction[g.idx(iz, ir)] = 1.0;
            }
        }
        assert!(column_reflect_supported(&g, &noz));
        // No solid at all: supported (the fill is a no-op).
        assert!(column_reflect_supported(&g, &SolidField::empty(g.clone())));
        // A floating disk above the wall run: two solid runs in its columns.
        let mut disks = noz.clone();
        for iz in 8..11usize {
            disks.fraction[g.idx(iz, 2)] = 1.0;
        }
        assert!(!column_reflect_supported(&g, &disks));
    }

    #[test]
    fn nozzle_gate_rejects_disk_array_and_baffles() {
        let g = Grid::uniform(60, 24, 0.1, 0.05);
        // 3x3 array of disconnected square "disks": every disk column has two
        // or more fluid runs. The old gate read this as a nozzle and seeded
        // 73.1% of its fluid cells at up to M 2.07.
        let mut disks = SolidField::empty(g.clone());
        for bz in 0..3usize {
            for br in 0..3usize {
                for iz in (10 + bz * 15)..(15 + bz * 15) {
                    for ir in (4 + br * 6)..(8 + br * 6) {
                        disks.fraction[g.idx(iz, ir)] = 1.0;
                    }
                }
            }
        }
        assert!(nozzle_profile(&g, &disks).is_none(), "disk array is not a nozzle");
        // Duct with thin baffles: solid top wall plus two one-cell baffles
        // hanging into the duct. Baffle columns have one fluid run below plus
        // the geometry converges-then-diverges in open radius — the old gate
        // passed it and seeded 98.4% of the duct at up to M 3.13.
        let mut baffled = SolidField::empty(g.clone());
        for iz in 0..g.nz {
            baffled.fraction[g.idx(iz, g.nr - 1)] = 1.0;
        }
        for &iz in &[20usize, 40] {
            for ir in 8..(g.nr - 1) {
                baffled.fraction[g.idx(iz, ir)] = 1.0;
            }
        }
        assert!(nozzle_profile(&g, &baffled).is_none(), "baffled duct is not a nozzle");
        // A mid-duct baffle detached from both walls (two fluid runs).
        let mut floating = SolidField::empty(g.clone());
        for iz in 0..g.nz {
            floating.fraction[g.idx(iz, g.nr - 1)] = 1.0;
        }
        for ir in 6..14usize {
            floating.fraction[g.idx(30, ir)] = 1.0;
        }
        assert!(nozzle_profile(&g, &floating).is_none(),
                "floating baffle is not a nozzle");
    }

    #[test]
    fn nozzle_gate_accepts_the_cd_nozzle() {
        let g = Grid::uniform(60, 24, 0.1, 0.05);
        let solid = cd_nozzle(&g, cd_wall);
        let p = nozzle_profile(&g, &solid).expect("CD nozzle must pass the gate");
        assert_eq!(p.lip, 40);
        // cd_wall's integer division keeps the wall at 8 through iz = 24, so
        // the flat throat's last argmin — its downstream end — is 24.
        assert_eq!(p.i_throat, 24, "last argmin of the flat throat");
        assert!(p.r_open[p.i_throat] < p.r_open[0]);
        assert!(p.r_open[p.i_throat] < p.r_open[p.lip]);
    }

    #[test]
    fn quasi1d_generic_fallback_freestream_only_on_the_isentrope() {
        let g = Grid::uniform(60, 24, 0.1, 0.05);
        let gm = 1.4f32;
        // Disk-array geometry so the gate fails.
        let mut disks = SolidField::empty(g.clone());
        for iz in 20..25usize {
            for ir in 6..10usize {
                disks.fraction[g.idx(iz, ir)] = 1.0;
            }
            for ir in 14..18usize {
                disks.fraction[g.idx(iz, ir)] = 1.0;
            }
        }
        // External-flow rig at M = 2: chamber = stagnation of the (1,1) stream,
        // ambient = the stream's static state. Fallback = uniform freestream.
        let m_inf = 2.0f64;
        let t0 = 1.0 + 0.5 * 0.4 * m_inf * m_inf;
        let chamber2 = Chamber { p0: t0.powf(1.4 / 0.4) as f32, t0: t0 as f32 };
        let amb2 = Ambient { p: 1.0, t: 1.0 };
        let mut u = vec![[0.0f32; 4]; g.glen()];
        quasi1d_init(&mut u, &g, &disks, &gas(), &chamber2, &amb2);
        let w = cons_to_prim(u[g.gidx(5, 5)], gm);
        let u_inf = m_inf * 1.4f64.sqrt();
        assert!((w[0] as f64 - 1.0).abs() < 1e-4, "rho {}", w[0]);
        assert!((w[1] as f64 - u_inf).abs() < 1e-4 * u_inf, "u {}", w[1]);
        assert!((w[3] as f64 - 1.0).abs() < 1e-4, "p {}", w[3]);
        // Rocket chamber over a low ambient: supersonic pressure ratio but the
        // ambient is NOT the chamber's stream (t off the isentrope) — ambient
        // at rest, not a M 3.2 blast through the disks.
        let amb = ambient(); // p 0.02, t 0.6; isentropic t at that p is 0.197
        let mut u = vec![[0.0f32; 4]; g.glen()];
        quasi1d_init(&mut u, &g, &disks, &gas(), &chamber(), &amb);
        assert_eq!(u[g.gidx(5, 5)], ambient_cons(&amb, gm));
        assert_eq!(u[g.gidx(50, 20)], ambient_cons(&amb, gm));
    }

    // ---- geometry change --------------------------------------------------

    /// f64 totals of mass, z-momentum, r-momentum, energy over the fluid
    /// cells of `solid`.
    fn totals(u: &[Cons], g: &Grid, solid: &SolidField) -> (f64, f64, f64, f64) {
        let mut t = [0.0f64; 4];
        for ir in 0..g.nr {
            for iz in 0..g.nz {
                if solid.is_solid(g.idx(iz, ir)) {
                    continue;
                }
                let c = u[g.gidx(iz as isize, ir as isize)];
                let vol = g.cell_vol(iz, ir);
                for k in 0..4 {
                    t[k] += c[k] as f64 * vol;
                }
            }
        }
        (t[0], t[1], t[2], t[3])
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
        let (m0, pz0, pr0, e0) = totals(&u, &g, &old);
        let mut ledger = FlipLedger::default();
        apply_geometry_change(&mut u, &g, &old, &new, &gas(), &ambient(), &mut ledger);
        let (m1, pz1, pr1, e1) = totals(&u, &g, &new);
        assert!(ledger.mass < 0.0, "closing cells must remove mass");
        assert!(ledger.momentum_z < 0.0, "closing cells moving in +z must remove z-momentum");
        assert!(ledger.momentum_r > 0.0, "closing cells moving in -r must remove negative r-momentum");
        assert!(((m1 - m0) - ledger.mass).abs() <= 1e-12 * m0, "T2 invariant (mass)");
        assert!(((pz1 - pz0) - ledger.momentum_z).abs() <= 1e-12 * pz0.abs().max(1.0),
                "T2 invariant (z-momentum)");
        assert!(((pr1 - pr0) - ledger.momentum_r).abs() <= 1e-12 * pr0.abs().max(1.0),
                "T2 invariant (r-momentum)");
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
        let (m0, pz0, pr0, e0) = totals(&u, &g, &old);
        let mut ledger = FlipLedger::default();
        apply_geometry_change(&mut u, &g, &old, &new, &gas(), &amb, &mut ledger);
        let (m1, pz1, pr1, e1) = totals(&u, &g, &new);
        assert!(ledger.mass > 0.0, "opening cells must add mass");
        assert_eq!(ledger.momentum_z, 0.0, "at-rest fills book zero z-momentum");
        assert_eq!(ledger.momentum_r, 0.0, "at-rest fills book zero r-momentum");
        assert!(((m1 - m0) - ledger.mass).abs() <= 1e-12 * m0.max(1.0), "T2 invariant (mass)");
        assert!(((pz1 - pz0) - ledger.momentum_z).abs() <= 1e-12 * pz0.abs().max(1.0),
                "T2 invariant (z-momentum)");
        assert!(((pr1 - pr0) - ledger.momentum_r).abs() <= 1e-12 * pr0.abs().max(1.0),
                "T2 invariant (r-momentum)");
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
    fn geometry_open_sealed_cavity_extrapolates_nearest_fluid_at_rest() {
        // A2: the sealed-cavity fallback is the NEAREST valid fluid cell's
        // (rho, p) at rest, not ambient — filling a void inside a hot body
        // with cold far-field ambient plants a spurious pressure jump.
        let g = Grid::uniform(10, 8, 0.1, 0.05);
        let gm = 1.4f32;
        let amb = ambient();
        let old = block_solid(&g, 3, 5, 2, 4); // 3x3 solid block
        let mut new = block_solid(&g, 3, 5, 2, 4);
        new.fraction[g.idx(4, 3)] = 0.0; // open the centre; ring stays solid
        let mut u = filled_grid(&g, |_, _| [0.9, 0.5, 0.1, 0.8], gm);
        let mut ledger = FlipLedger::default();
        apply_geometry_change(&mut u, &g, &old, &new, &gas(), &amb, &mut ledger);
        // Every valid fluid cell holds (0.9, _, _, 0.8): the cavity must too,
        // at rest.
        let expect = prim_to_cons([0.9, 0.0, 0.0, 0.8], gm);
        assert_eq!(u[g.gidx(4, 3)], expect);
        let vol = g.cell_vol(4, 3);
        // 0.9 is not exact in f32; compare against the f32-rounded value.
        assert!((ledger.mass - 0.9f32 as f64 * vol).abs() <= 1e-12 * vol,
                "cavity mass on the ledger: {} vs {}", ledger.mass, 0.9f32 as f64 * vol);
        assert_eq!(ledger.momentum_z, 0.0);
        assert_eq!(ledger.momentum_r, 0.0);
    }

    #[test]
    fn geometry_open_large_block_refills_completely() {
        // A2: the BFS was capped at 8 passes — erasing a 40x40 obstacle left
        // 48% of it at ambient, 80x80 left 80%. It now runs to completion:
        // every opened cell must be reached (no cell keeps the ambient the
        // closure pass wrote into solid cells) and the T2 invariant holds.
        let g = Grid::uniform(48, 48, 0.05, 0.05);
        let gm = 1.4f32;
        let amb = ambient();
        let old = block_solid(&g, 4, 43, 4, 43); // 40x40 block
        let new = SolidField::empty(g.clone());
        let mut u = filled_grid(&g, |iz, ir| {
            [1.0 + 0.001 * (ir + iz) as f32, 0.3, -0.1, 1.0 + 0.0005 * iz as f32]
        }, gm);
        for ir in 4..=43usize {
            for iz in 4..=43usize {
                u[g.gidx(iz as isize, ir as isize)] = ambient_cons(&amb, gm);
            }
        }
        let (m0, pz0, pr0, e0) = totals(&u, &g, &old);
        let mut ledger = FlipLedger::default();
        apply_geometry_change(&mut u, &g, &old, &new, &gas(), &amb, &mut ledger);
        let (m1, pz1, pr1, e1) = totals(&u, &g, &new);
        // No opened cell may sit at ambient: the shallowest interior fluid
        // pressure around the block is ~1.0, ambient is 0.02, and the BFS
        // mean can never cross below the neighbourhood minimum.
        for ir in 4..=43usize {
            for iz in 4..=43usize {
                let c = u[g.gidx(iz as isize, ir as isize)];
                let w = cons_to_prim(c, gm);
                assert!(w[3] > 0.5, "({iz}, {ir}) left near ambient: p = {}", w[3]);
                assert_eq!(c[1], 0.0, "({iz}, {ir}) not at rest");
                assert_eq!(c[2], 0.0, "({iz}, {ir}) not at rest");
            }
        }
        assert!(((m1 - m0) - ledger.mass).abs() <= 1e-12 * m0.max(1.0), "T2 invariant (mass)");
        assert!(((pz1 - pz0) - ledger.momentum_z).abs() <= 1e-12 * pz0.abs().max(1.0),
                "T2 invariant (z-momentum)");
        assert!(((pr1 - pr0) - ledger.momentum_r).abs() <= 1e-12 * pr0.abs().max(1.0),
                "T2 invariant (r-momentum)");
        assert!(((e1 - e0) - ledger.energy).abs() <= 1e-12 * e0.max(1.0), "T2 invariant (energy)");
    }
}
