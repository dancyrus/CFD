//! Acceptance ladder skeleton — T1, T4 and the well-balanced test, written
//! against the `Solver` trait BEFORE the solver exists so session A has a
//! red-to-green target instead of its own definition of done.
//!
//! All three are #[ignore]d until kernel.rs and physics.rs are implemented:
//!
//!     cargo test -p cfd-core -- --ignored t1_freestream t4_sod well_balanced
//!
//! Verification tests hard-code gamma = 1.4 — the exact references only exist
//! there. References: docs/physics-reference.md §12.

use std::sync::Arc;

use cfd_contract::{
    Ambient, Chamber, FieldKind, GasModel, Geometry, Grid, Numerics, RefScales,
    SolidField, SolveSetup, Solver,
};
use cfd_core::EulerSolver;

/// Identity scales: SI == non-dimensional, so Snapshot fields can be asserted
/// against non-dimensional literals directly.
fn identity_refs() -> RefScales {
    RefScales { l_m: 1.0, p_pa: 1.0, rho_kg_m3: 1.0, u_m_s: 1.0, t_k: 1.0, time_s: 1.0 }
}

fn numerics_for_tests(geometry: Geometry) -> Numerics {
    Numerics { geometry, quasi1d_init: false, sponge_cells: 0, ..Numerics::default() }
}

// The exact Sod star state, independently recomputed (physics-reference §12).
const P_STAR: f64 = 0.3031301781;
const U_STAR: f64 = 0.9274526200;
const RHO_STAR_L: f64 = 0.4263194282;
const RHO_STAR_R: f64 = 0.2655737117;
const SHOCK_SPEED: f64 = 1.7521557320;

/// T1 — free-stream preservation. A planar uniform stream (rho=1, u=2, p=1)
/// with the chamber set to the stream's stagnation state must pass through
/// unchanged: max|rho-1|, |u-2|, |p-1| <= 1e-5 and max|rho*v| <= 1e-6 after
/// 200 steps. NOTE: this cannot catch a wrong axisymmetric source — the
/// source is identically zero here. T7 is what validates that machinery.
#[test]
#[ignore = "needs kernel.rs + physics.rs (sessions A and B)"]
fn t1_freestream() {
    let gamma = 1.4f64;
    let grid = Grid { nz: 64, nr: 8, dz: 0.05, dr: 0.05 };
    let (rho, u, p) = (1.0f64, 2.0f64, 1.0f64);
    // Stagnation state of the stream: T0 = T*(1 + (g-1)/2 M^2), isentropic p0.
    let t = p / rho;
    let m2 = u * u / (gamma * t);
    let t0 = t * (1.0 + 0.5 * (gamma - 1.0) * m2);
    let p0 = p * (t0 / t).powf(gamma / (gamma - 1.0));
    let setup = SolveSetup {
        grid,
        solid: Arc::new(SolidField::empty(grid)),
        gas: GasModel { gamma: gamma as f32, r_specific_si: 287.0 },
        chamber: Chamber { p0: p0 as f32, t0: t0 as f32 },
        ambient: Ambient { p: p as f32, t: t as f32 },
        numerics: numerics_for_tests(Geometry::Planar),
        refs: identity_refs(),
    };
    let mut s = EulerSolver::new(setup).unwrap();
    s.set_initial(|_, _| [rho as f32, u as f32, 0.0, p as f32]);
    for _ in 0..200 { s.step().unwrap(); }
    let snap = s.snapshot();
    let (mut e_rho, mut e_u, mut e_p, mut e_rv) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for ir in 0..grid.nr {
        for iz in 0..grid.nz {
            let d = snap.sample(FieldKind::Density, iz, ir) as f64;
            e_rho = e_rho.max((d - rho).abs());
            e_u = e_u.max((snap.sample(FieldKind::VelocityZ, iz, ir) as f64 - u).abs());
            e_p = e_p.max((snap.sample(FieldKind::Pressure, iz, ir) as f64 - p).abs());
            e_rv = e_rv.max((d * snap.sample(FieldKind::VelocityR, iz, ir) as f64).abs());
        }
    }
    assert!(e_rho <= 1e-5, "max|rho-1| = {e_rho}");
    assert!(e_u <= 1e-5, "max|u-2| = {e_u}");
    assert!(e_p <= 1e-5, "max|p-1| = {e_p}");
    assert!(e_rv <= 1e-6, "max|rho*v| = {e_rv}");
}

/// Exact Sod density at (x, t), diaphragm at x = 0.5. Self-similar solution
/// assembled from the recomputed star state.
fn sod_exact_rho(x: f64, t: f64) -> f64 {
    let g = 1.4f64;
    let (rho_l, p_l) = (1.0, 1.0);
    let rho_r = 0.125;
    let a_l = (g * p_l / rho_l).sqrt();
    let a_star_l = a_l * (P_STAR / p_l).powf((g - 1.0) / (2.0 * g));
    let xi = (x - 0.5) / t;
    if xi < -a_l {
        rho_l
    } else if xi < U_STAR - a_star_l {
        // Left rarefaction fan.
        rho_l * (2.0 / (g + 1.0) - (g - 1.0) / ((g + 1.0) * a_l) * xi).powf(2.0 / (g - 1.0))
    } else if xi < U_STAR {
        RHO_STAR_L
    } else if xi < SHOCK_SPEED {
        RHO_STAR_R
    } else {
        rho_r
    }
}

/// T4 — Sod on a 1-cell-tall planar strip. N = 200, gamma = 1.4, run to
/// t = 0.2. L1(rho) <= 6.0e-3: correct second order measures 2.4-4.1e-3,
/// first order 1.32e-2 — the threshold sits in the gap and catches a solver
/// that is first order and calls itself MUSCL. Also: shock front within
/// +/-1.5 dz, max rho <= 1.001, max|rho*v| <= 1e-8 (transverse flux leakage).
#[test]
#[ignore = "needs kernel.rs + physics.rs (sessions A and B)"]
fn t4_sod() {
    let n = 200usize;
    let grid = Grid { nz: n, nr: 2, dz: 1.0 / n as f32, dr: 1.0 / n as f32 };
    // Row 1 is solid: the strip is one fluid cell tall, radially inert.
    // (This is why Geometry::Planar and the grid types must allow it.)
    let mut solid = SolidField::empty(grid);
    for iz in 0..n { solid.fraction[grid.idx(iz, 1)] = 1.0; }
    // Sod-left (1, 0, 1) at rest IS the chamber state, so the stagnation
    // inlet reproduces it exactly; the right state is ambient at p = 0.1.
    let setup = SolveSetup {
        grid,
        solid: Arc::new(solid),
        gas: GasModel { gamma: 1.4, r_specific_si: 287.0 },
        chamber: Chamber { p0: 1.0, t0: 1.0 },
        ambient: Ambient { p: 0.1, t: 0.8 },
        numerics: numerics_for_tests(Geometry::Planar),
        refs: identity_refs(),
    };
    let mut s = EulerSolver::new(setup).unwrap();
    s.set_initial(|iz, _| {
        let x = (iz as f32 + 0.5) / n as f32;
        if x < 0.5 { [1.0, 0.0, 0.0, 1.0] } else { [0.125, 0.0, 0.0, 0.1] }
    });
    let mut info = s.step().unwrap();
    while info.time < 0.2 { info = s.step().unwrap(); }
    let t = info.time; // compare at the actual final time, not nominal 0.2
    let snap = s.snapshot();

    let mut l1 = 0.0f64;
    let mut rho_max = 0.0f64;
    let mut rv_max = 0.0f64;
    let mut shock_x = 0.0f64;
    let mid = 0.5 * (RHO_STAR_R + 0.125);
    for iz in 0..n {
        let x = (iz as f64 + 0.5) / n as f64;
        let rho = snap.sample(FieldKind::Density, iz, 0) as f64;
        l1 += (rho - sod_exact_rho(x, t)).abs() * grid.dz as f64;
        rho_max = rho_max.max(rho);
        rv_max = rv_max.max((rho * snap.sample(FieldKind::VelocityR, iz, 0) as f64).abs());
        if rho > mid { shock_x = x; } // last cell still above the mid-density
    }
    let shock_exact = 0.5 + SHOCK_SPEED * t;
    assert!(l1 <= 6.0e-3, "L1(rho) = {l1:.4e} (2nd order: 2.4-4.1e-3, 1st: 1.32e-2)");
    assert!((shock_x - shock_exact).abs() <= 1.5 * grid.dz as f64,
            "shock at {shock_x}, exact {shock_exact}");
    assert!(rho_max <= 1.001, "max rho = {rho_max}");
    assert!(rv_max <= 1e-8, "max|rho*v| = {rv_max} — transverse flux leakage");
}

/// Well-balanced — a closed box of quiescent uniform-pressure gas in
/// axisymmetric mode. For uniform p the source p_j*dr cancels the radial flux
/// difference p*(r_{j+1/2} - r_{j-1/2}) EXACTLY at every row including 0 —
/// but only if sweep_r keeps both inside the single bracket
/// (docs/physics-reference.md §1). The state must stay at machine zero.
#[test]
#[ignore = "needs kernel.rs + physics.rs (sessions A and B)"]
fn well_balanced() {
    let grid = Grid { nz: 32, nr: 32, dz: 0.1, dr: 0.1 };
    // Solid walls on both z ends and the outer radius: no open boundaries,
    // so any drift is the interior discretization's own.
    let mut solid = SolidField::empty(grid);
    for ir in 0..grid.nr {
        solid.fraction[grid.idx(0, ir)] = 1.0;
        solid.fraction[grid.idx(grid.nz - 1, ir)] = 1.0;
    }
    for iz in 0..grid.nz { solid.fraction[grid.idx(iz, grid.nr - 1)] = 1.0; }
    let setup = SolveSetup {
        grid,
        solid: Arc::new(solid),
        gas: GasModel { gamma: 1.4, r_specific_si: 287.0 },
        chamber: Chamber { p0: 1.0, t0: 1.0 },
        ambient: Ambient { p: 1.0, t: 1.0 },
        numerics: numerics_for_tests(Geometry::Axisymmetric),
        refs: identity_refs(),
    };
    let mut s = EulerSolver::new(setup).unwrap();
    s.set_initial(|_, _| [1.0, 0.0, 0.0, 1.0]);
    for _ in 0..100 { s.step().unwrap(); }
    let snap = s.snapshot();
    let mut err = 0.0f64;
    for ir in 0..grid.nr {
        for iz in 0..grid.nz {
            if solid_at(&snap, iz, ir) { continue; }
            err = err.max((snap.sample(FieldKind::Density, iz, ir) as f64 - 1.0).abs());
            err = err.max((snap.sample(FieldKind::Pressure, iz, ir) as f64 - 1.0).abs());
            err = err.max((snap.sample(FieldKind::VelocityZ, iz, ir) as f64).abs());
            err = err.max((snap.sample(FieldKind::VelocityR, iz, ir) as f64).abs());
        }
    }
    assert!(err <= 1e-7, "quiescent uniform state drifted by {err:.3e}");
}

fn solid_at(snap: &cfd_contract::Snapshot, iz: usize, ir: usize) -> bool {
    snap.solid.is_solid(snap.grid.idx(iz, ir))
}
