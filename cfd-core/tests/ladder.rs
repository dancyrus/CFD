//! Acceptance ladder rungs T2, T3, T5, T6, T7, T8 (T0 lives in cfd-geom,
//! T1/T4/well-balanced in tests/acceptance.rs). References and pass bands:
//! docs/physics-reference.md §12. Every test prints its measured numbers so
//! the integrator can report actuals, not just pass/fail.
//!
//! All are #[ignore]d — they run seconds to a minute each:
//!     cargo test -p cfd-core --test ladder -- --include-ignored --nocapture

use std::sync::Arc;

use cfd_contract::{
    Ambient, Chamber, FieldKind, GasModel, Geometry, Grid, Numerics, RefScales, Snapshot,
    SolidField, SolveSetup, Solver,
};
use cfd_core::{physics, EulerSolver};

fn identity_refs() -> RefScales {
    RefScales { l_m: 1.0, p_pa: 1.0, rho_kg_m3: 1.0, u_m_s: 1.0, t_k: 1.0, time_s: 1.0 }
}

fn test_numerics(geometry: Geometry) -> Numerics {
    Numerics { geometry, quasi1d_init: false, sponge_cells: 0, ..Numerics::default() }
}

/// Chamber = the stagnation state of a (rho=1, p=1, T=1) stream at Mach m, so
/// the stagnation inlet reproduces the free stream exactly (verified by T1).
fn freestream_chamber(m: f64, gamma: f64) -> Chamber {
    let t0 = 1.0 + 0.5 * (gamma - 1.0) * m * m;
    Chamber { p0: t0.powf(gamma / (gamma - 1.0)) as f32, t0: t0 as f32 }
}

fn run_to_time(s: &mut EulerSolver, t_end: f64, max_steps: u64) -> cfd_contract::StepInfo {
    let mut info = s.step().unwrap();
    let mut n = 1;
    while info.time < t_end && n < max_steps {
        info = s.step().unwrap();
        n += 1;
    }
    assert!(info.time >= t_end, "ran out of steps at t = {}", info.time);
    info
}

/// f64 totals of mass and energy over the interior, from a snapshot with
/// identity reference scales (fields are non-dimensional). Solid cells carry
/// zeros in every field, so they contribute nothing.
fn totals(snap: &Snapshot, gamma: f64) -> (f64, f64) {
    let g = snap.grid;
    let (mut mass, mut energy) = (0.0f64, 0.0f64);
    for ir in 0..g.nr {
        let vol = g.cell_vol(ir);
        for iz in 0..g.nz {
            let rho = snap.sample(FieldKind::Density, iz, ir) as f64;
            let p = snap.sample(FieldKind::Pressure, iz, ir) as f64;
            let uz = snap.sample(FieldKind::VelocityZ, iz, ir) as f64;
            let ur = snap.sample(FieldKind::VelocityR, iz, ir) as f64;
            mass += rho * vol;
            energy += (p / (gamma - 1.0) + 0.5 * rho * (uz * uz + ur * ur)) * vol;
        }
    }
    (mass, energy)
}

/// T2 — conservation drift. A closed solid box of axisymmetric gas with a
/// smooth pressure/density blob sloshing at rest for 1000 steps; mass and
/// energy drift <= 2e-6 relative, all diagnostics in f64. The flip-ledger
/// accounting for mid-run geometry edits is covered by unit tests in
/// physics.rs (geometry_close_books_removed_mass_on_the_ledger and friends);
/// the in-loop |Δmass − ledger.mass| form needs EulerSolver's private ledger,
/// which the frozen step.rs does not expose.
#[test]
#[ignore = "ladder: run with --include-ignored"]
fn t2_conservation_drift() {
    let gamma = 1.4f64;
    let grid = Grid { nz: 48, nr: 48, dz: 0.1, dr: 0.1 };
    let mut solid = SolidField::empty(grid);
    for ir in 0..grid.nr {
        solid.fraction[grid.idx(0, ir)] = 1.0;
        solid.fraction[grid.idx(grid.nz - 1, ir)] = 1.0;
    }
    for iz in 0..grid.nz {
        solid.fraction[grid.idx(iz, grid.nr - 1)] = 1.0;
    }
    let setup = SolveSetup {
        grid,
        solid: Arc::new(solid),
        gas: GasModel { gamma: gamma as f32, r_specific_si: 287.0 },
        chamber: Chamber { p0: 1.0, t0: 1.0 },
        ambient: Ambient { p: 1.0, t: 1.0 },
        numerics: test_numerics(Geometry::Axisymmetric),
        refs: identity_refs(),
    };
    let mut s = EulerSolver::new(setup).unwrap();
    s.set_initial(|iz, ir| {
        let z = (iz as f32 + 0.5) * 0.1;
        let r = (ir as f32 + 0.5) * 0.1;
        let bump = 0.25 * (-((z - 2.4).powi(2) + (r - 2.0).powi(2)) / 0.36).exp();
        [1.0 + bump, 0.0, 0.0, 1.0 + bump] // T = 1 uniform
    });
    let (m0, e0) = totals(&s.snapshot(), gamma);
    let mut info = s.step().unwrap();
    for _ in 0..999 {
        info = s.step().unwrap();
    }
    let (m1, e1) = totals(&s.snapshot(), gamma);
    let dm = ((m1 - m0) / m0).abs();
    let de = ((e1 - e0) / e0).abs();
    println!("T2: mass drift {dm:.3e}, energy drift {de:.3e} (pass <= 2e-6), floors {}",
             info.floor_activations);
    assert!(dm <= 2e-6, "mass drift {dm:.3e}");
    assert!(de <= 2e-6, "energy drift {de:.3e}");
    assert_eq!(info.floor_activations, 0);
}

/// One T3 run: an entropy wave (smooth density bump, uniform u = 2, p = 1)
/// advected to t ~ 0.1 on an N-cell planar strip. Returns L1(rho) vs the
/// exactly shifted profile at the run's own final time.
fn t3_error(n: usize, limiter: cfd_contract::Limiter) -> f64 {
    let gamma = 1.4f64;
    let grid = Grid { nz: n, nr: 2, dz: 1.0 / n as f32, dr: 1.0 / n as f32 };
    let mut solid = SolidField::empty(grid);
    for iz in 0..n {
        solid.fraction[grid.idx(iz, 1)] = 1.0;
    }
    let bump = |x: f64| 1.0 + 0.2 * (-((x - 0.35) / 0.08).powi(2)).exp();
    let m = 2.0 / (gamma).sqrt(); // u = 2 stream of the (1, 2, 1) state
    let setup = SolveSetup {
        grid,
        solid: Arc::new(solid),
        gas: GasModel { gamma: gamma as f32, r_specific_si: 287.0 },
        chamber: freestream_chamber(m, gamma),
        ambient: Ambient { p: 1.0, t: 1.0 },
        numerics: Numerics { limiter, ..test_numerics(Geometry::Planar) },
        refs: identity_refs(),
    };
    let mut s = EulerSolver::new(setup).unwrap();
    s.set_initial(|iz, _| {
        let x = (iz as f64 + 0.5) / n as f64;
        [bump(x) as f32, 2.0, 0.0, 1.0]
    });
    let info = run_to_time(&mut s, 0.1, 10_000);
    let snap = s.snapshot();
    let mut l1 = 0.0f64;
    for iz in 0..n {
        let x = (iz as f64 + 0.5) / n as f64;
        let rho = snap.sample(FieldKind::Density, iz, 0) as f64;
        l1 += (rho - bump(x - 2.0 * info.time)).abs() / n as f64;
    }
    l1
}

/// T3 — order of accuracy on smooth advection. Two thresholds, because a
/// limited scheme cannot hit 2.0 and "1.6" alone is ambiguous between a
/// healthy limiter and broken reconstruction.
#[test]
#[ignore = "ladder: run with --include-ignored"]
fn t3_order_of_accuracy() {
    let e100 = t3_error(100, cfd_contract::Limiter::None);
    let e200 = t3_error(200, cfd_contract::Limiter::None);
    let order_unlimited = (e100 / e200).log2();
    let e100 = t3_error(100, cfd_contract::Limiter::Minmod);
    let e200 = t3_error(200, cfd_contract::Limiter::Minmod);
    let e400 = t3_error(400, cfd_contract::Limiter::Minmod);
    println!("T3 minmod pairs: 100/200 -> {:.3}, 200/400 -> {:.3}",
             (e100 / e200).log2(), (e200 / e400).log2());
    let order_limited = (e200 / e400).log2();
    println!("T3: order unlimited {order_unlimited:.3} (pass >= 1.90), \
              limited {order_limited:.3} (pass >= 1.50)");
    assert!(order_unlimited >= 1.90, "unlimited order {order_unlimited:.3}");
    assert!(order_limited >= 1.50, "limited order {order_limited:.3}");
}

/// T5a — positivity: Toro test 2, the double rarefaction with a near-vacuum
/// star state (p* = 1.894e-3). Floor activations = 0 is the pass; nonzero is
/// a hard stop, not a soft failure.
#[test]
#[ignore = "ladder: run with --include-ignored"]
fn t5_positivity_toro2() {
    let n = 200usize;
    let grid = Grid { nz: n, nr: 2, dz: 1.0 / n as f32, dr: 1.0 / n as f32 };
    let mut solid = SolidField::empty(grid);
    for iz in 0..n {
        solid.fraction[grid.idx(iz, 1)] = 1.0;
    }
    let setup = SolveSetup {
        grid,
        solid: Arc::new(solid),
        gas: GasModel { gamma: 1.4, r_specific_si: 287.0 },
        chamber: Chamber { p0: 1.0, t0: 1.0 },
        ambient: Ambient { p: 0.4, t: 0.4 },
        numerics: test_numerics(Geometry::Planar),
        refs: identity_refs(),
    };
    let mut s = EulerSolver::new(setup).unwrap();
    s.set_initial(|iz, _| {
        let x = (iz as f32 + 0.5) / n as f32;
        if x < 0.5 { [1.0, -2.0, 0.0, 0.4] } else { [1.0, 2.0, 0.0, 0.4] }
    });
    let info = run_to_time(&mut s, 0.15, 10_000);
    let snap = s.snapshot();
    let p_min = (0..n)
        .map(|iz| snap.sample(FieldKind::Pressure, iz, 0) as f64)
        .fold(f64::INFINITY, f64::min);
    println!("T5 (Toro 2): floors {} (pass = 0), min p {p_min:.4e} (exact p* 1.894e-3)",
             info.floor_activations);
    assert_eq!(info.floor_activations, 0, "floor activations must be exactly zero");
    assert!(p_min > 0.0 && p_min.is_finite());
}

/// The demo nozzle (gamma 1.24, eps 8, conical 15 deg) on the interactive
/// grid, via the real cfd-geom contour + rasterizer pipeline.
fn nozzle_setup(ambient_p_pa: f64, ambient_t_k: f64) -> SolveSetup {
    let gas = GasModel { gamma: 1.24, r_specific_si: 378.0 };
    let refs = RefScales::from_chamber(0.05, 5.0e6, 3200.0, &gas);
    let grid = Grid { nz: 320, nr: 200, dz: 0.1449, dr: 0.05 };
    let spec = cfd_geom::NozzleSpec {
        throat_radius_m: 0.05,
        area_ratio: 8.0,
        contraction_ratio: 4.0,
        converge_half_angle_deg: 30.0,
        throat_arc_up: 1.5,
        throat_arc_down: 0.382,
        contour: cfd_geom::ContourKind::Conical { half_angle_deg: 15.0 },
    };
    let wall = cfd_geom::generate_contour(&spec, 512).unwrap();
    let solid = cfd_geom::rasterize(&wall, &grid, &refs).unwrap();
    SolveSetup {
        grid,
        solid: Arc::new(solid),
        gas,
        chamber: Chamber { p0: 1.0, t0: 1.0 },
        ambient: Ambient {
            p: (ambient_p_pa / refs.p_pa) as f32,
            t: (ambient_t_k / refs.t_k) as f32,
        },
        numerics: Numerics::default(), // quasi-1D init on, sponge 24
        refs,
    }
}

/// T5b — positivity at the vacuum end of the altitude slider: the demo nozzle
/// starting into a 50 km ambient (p_a = 76 Pa). The startup expansion into
/// near-vacuum is where the floors trip if the two-level fallback is wrong.
#[test]
#[ignore = "ladder: run with --include-ignored"]
fn t5_positivity_vacuum_nozzle() {
    let setup = nozzle_setup(76.0, 270.0);
    let mut s = EulerSolver::new(setup).unwrap();
    let mut info = s.step().unwrap();
    for _ in 0..1499 {
        info = s.step().unwrap();
    }
    println!("T5 (vacuum nozzle, 1500 steps): floors {} (pass = 0)", info.floor_activations);
    assert_eq!(info.floor_activations, 0, "floor activations must be exactly zero");
}

/// Least-squares shock angle: for each column in the fit window, scan from the
/// top down to the first pressure crossing of `threshold` (linear sub-cell
/// interpolation), then fit r = a + b*z and return atan(b) in degrees.
fn fitted_shock_angle_deg(
    snap: &Snapshot, iz_lo: usize, iz_hi: usize, threshold: f64,
) -> f64 {
    let g = snap.grid;
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for iz in iz_lo..=iz_hi {
        let mut crossing = None;
        for ir in (1..g.nr).rev() {
            let p_hi = snap.sample(FieldKind::Pressure, iz, ir) as f64;
            let p_lo = snap.sample(FieldKind::Pressure, iz, ir - 1) as f64;
            if p_hi < threshold && p_lo >= threshold {
                let t = (threshold - p_hi) / (p_lo - p_hi);
                crossing = Some(g.r_center(ir) as f64 - t * g.dr as f64);
                break;
            }
        }
        if let Some(r) = crossing {
            pts.push(((iz as f64 + 0.5) * g.dz as f64, r));
        }
    }
    assert!(pts.len() > (iz_hi - iz_lo) / 2, "shock not found in the fit window");
    let n = pts.len() as f64;
    let (sx, sy): (f64, f64) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    let (sxx, sxy): (f64, f64) = pts.iter().fold((0.0, 0.0), |a, p| {
        (a.0 + p.0 * p.0, a.1 + p.0 * p.1)
    });
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    slope.atan().to_degrees()
}

/// T6 — oblique shock off a 15 deg wedge at M = 2, gamma = 1.4, planar.
/// Reference beta = 45.34362 deg; fit over x in [x0+60dz, x0+150dz] — the
/// first 60 cells are excluded because the staircase leading edge is worst
/// there and including it fails a correct solver.
#[test]
#[ignore = "ladder: run with --include-ignored"]
fn t6_wedge_theta_beta_m() {
    let gamma = 1.4f64;
    let m_inf = 2.0f64;
    let grid = Grid { nz: 300, nr: 220, dz: 0.01, dr: 0.01 };
    let z0 = 0.5f64; // wedge tip
    let tan_w = 15.0f64.to_radians().tan();
    let mut solid = SolidField::empty(grid);
    for iz in 0..grid.nz {
        let z = (iz as f64 + 0.5) * grid.dz as f64;
        if z <= z0 { continue; }
        for ir in 0..grid.nr {
            let r = grid.r_center(ir) as f64;
            // Exact sub-cell fraction of the wedge in this cell is overkill;
            // centre sampling of a straight ramp gives the same staircase the
            // reference tolerances (+/-1.5 deg) were set for.
            if r < (z - z0) * tan_w {
                solid.fraction[grid.idx(iz, ir)] = 1.0;
            }
        }
    }
    let u_inf = m_inf * gamma.sqrt();
    let setup = SolveSetup {
        grid,
        solid: Arc::new(solid),
        gas: GasModel { gamma: gamma as f32, r_specific_si: 287.0 },
        chamber: freestream_chamber(m_inf, gamma),
        ambient: Ambient { p: 1.0, t: 1.0 },
        numerics: test_numerics(Geometry::Planar),
        refs: identity_refs(),
    };
    let mut s = EulerSolver::new(setup).unwrap();
    s.set_initial(|_, _| [1.0, u_inf as f32, 0.0, 1.0]);
    run_to_time(&mut s, 4.0, 20_000);
    let snap = s.snapshot();
    // p2/p1 = 2.1946; threshold midway between states.
    let iz_lo = (z0 / grid.dz as f64) as usize + 60;
    let iz_hi = (z0 / grid.dz as f64) as usize + 150;
    let beta = fitted_shock_angle_deg(&snap, iz_lo, iz_hi, 1.5);
    // Post-shock pressure, sampled just above the wedge surface mid-window.
    let mut p2 = 0.0f64;
    let mut np = 0usize;
    for iz in iz_lo..=iz_hi {
        let z = (iz as f64 + 0.5) * grid.dz as f64;
        let ir_surf = ((z - z0) * tan_w / grid.dr as f64).ceil() as usize + 1;
        p2 += snap.sample(FieldKind::Pressure, iz, ir_surf) as f64;
        np += 1;
    }
    p2 /= np as f64;
    println!("T6: beta {beta:.3} deg (ref 45.344 +/- 1.5), p2/p1 {p2:.4} (ref 2.1947)");
    assert!((beta - 45.34362).abs() <= 1.5, "beta = {beta:.3} deg");
}

/// T7 — cone vs Taylor–Maccoll, M = 2.35, theta_c = 10 deg, gamma = 1.4,
/// axisymmetric. References (independently recomputed): beta = 26.736718 deg
/// +/- 1.5, surface p/p_inf = 1.373936 +/- 8%, surface M = 2.146831 +/- 5%.
/// The NPARC archive's p3/p1 = 1.4234 is wrong for a 10 deg cone (it matches
/// 10.791 deg, and the archive's own beta and p are mutually inconsistent);
/// do NOT "fix" the reference back to it. This is the only test that validates
/// the axisymmetric source against a nontrivial exact solution — T1 cannot
/// catch a wrong source, it is identically zero there.
#[test]
#[ignore = "ladder: run with --include-ignored"]
fn t7_cone_taylor_maccoll() {
    let gamma = 1.4f64;
    let m_inf = 2.35f64;
    let grid = Grid { nz: 300, nr: 220, dz: 0.01, dr: 0.01 };
    let z0 = 0.5f64; // cone apex, on the axis
    let tan_c = 10.0f64.to_radians().tan();
    let mut solid = SolidField::empty(grid);
    for iz in 0..grid.nz {
        let z = (iz as f64 + 0.5) * grid.dz as f64;
        if z <= z0 { continue; }
        for ir in 0..grid.nr {
            if (grid.r_center(ir) as f64) < (z - z0) * tan_c {
                solid.fraction[grid.idx(iz, ir)] = 1.0;
            }
        }
    }
    let u_inf = m_inf * gamma.sqrt();
    let setup = SolveSetup {
        grid,
        solid: Arc::new(solid),
        gas: GasModel { gamma: gamma as f32, r_specific_si: 287.0 },
        chamber: freestream_chamber(m_inf, gamma),
        ambient: Ambient { p: 1.0, t: 1.0 },
        numerics: test_numerics(Geometry::Axisymmetric),
        refs: identity_refs(),
    };
    let mut s = EulerSolver::new(setup).unwrap();
    s.set_initial(|_, _| [1.0, u_inf as f32, 0.0, 1.0]);
    run_to_time(&mut s, 4.0, 20_000);
    let snap = s.snapshot();

    // Conical shock is weak: immediately behind it p/p_inf ~ 1.14. Threshold
    // sits between the freestream and that jump.
    let iz_lo = (z0 / grid.dz as f64) as usize + 60;
    let iz_hi = (z0 / grid.dz as f64) as usize + 150;
    let beta = fitted_shock_angle_deg(&snap, iz_lo, iz_hi, 1.05);

    // Surface values. Pressure is sampled at the first fluid cell — it is
    // nearly constant across the wall band (normal momentum balance). Mach is
    // sampled 8 cells above the surface: the staircase wall produces a ~6-cell
    // numerical entropy layer (measured in diag.rs — M rises 0.2 -> 2.13
    // across it and plateaus at the Taylor-Maccoll value), and inside that
    // layer surface Mach is a property of the wall artifact, not of the cone
    // flow the reference validates. Still well inside the shock layer for the
    // whole fit window (ray angle <= 17 deg vs beta = 26.7 deg).
    let (mut p_surf, mut m_surf) = (0.0f64, 0.0f64);
    let mut np = 0usize;
    for iz in iz_lo..=iz_hi {
        let z = (iz as f64 + 0.5) * grid.dz as f64;
        let ir_surf = ((z - z0) * tan_c / grid.dr as f64).ceil() as usize + 1;
        p_surf += snap.sample(FieldKind::Pressure, iz, ir_surf) as f64;
        m_surf += snap.sample(FieldKind::Mach, iz, ir_surf + 7) as f64;
        np += 1;
    }
    p_surf /= np as f64;
    m_surf /= np as f64;

    // Axis cleanliness upstream of the apex: rows 0 and 1, z < z0.
    let mut v_max = 0.0f64;
    for iz in 0..(z0 / grid.dz as f64) as usize {
        for ir in 0..2 {
            v_max = v_max.max((snap.sample(FieldKind::VelocityR, iz, ir) as f64).abs());
        }
    }
    println!("T7: beta {beta:.3} deg (ref 26.737 +/- 1.5), surface p/p_inf {p_surf:.4} \
              (ref 1.3739 +/- 8%), surface M {m_surf:.4} (ref 2.1468 +/- 5%), \
              upstream axis |v|/u_inf {:.2e} (pass <= 1e-3)", v_max / u_inf);
    assert!((beta - 26.736718).abs() <= 1.5, "beta = {beta:.3} deg");
    assert!((p_surf - 1.373936).abs() / 1.373936 <= 0.08, "surface p/p_inf = {p_surf:.4}");
    assert!((m_surf - 2.146831).abs() / 2.146831 <= 0.05, "surface M = {m_surf:.4}");
    assert!(v_max / u_inf <= 1e-3, "upstream axis |v|/u_inf = {:.2e}", v_max / u_inf);
}

/// T8 — the demo nozzle vs quasi-1D isentropic theory, references recomputed
/// from case.gas.gamma. mdot/mdot_ideal in [0.94, 1.00] (curved sonic line:
/// a correct 2D solver runs 0.3-1.0% BELOW ideal; above it is a bug);
/// area-averaged exit Mach within 4% of the isentropic value; C_f at
/// 0.975-0.995 of ideal (divergence loss is the physics being tested for).
#[test]
#[ignore = "ladder: run with --include-ignored"]
fn t8_nozzle_vs_isentropic() {
    let setup = nozzle_setup(101_325.0, 288.15);
    let gas = setup.gas;
    let refs = setup.refs;
    let p_a = setup.ambient.p as f64;
    let mut s = EulerSolver::new(setup).unwrap();
    let mut info = s.step().unwrap();
    let mut steps = 1u64;
    while !info.converged && steps < 12_000 {
        info = s.step().unwrap();
        steps += 1;
    }
    let r = s.report();
    assert_eq!(info.floor_activations, 0, "floors nonzero: every number is un-auditable");

    // Quasi-1D ideals at the spec area ratio, in f64 from the gas model.
    let g = gas.gamma as f64;
    let (p0, t0, rt) = (refs.p_pa, refs.t_k, refs.l_m);
    let a_t = std::f64::consts::PI * rt * rt;
    let a_e = a_t * 8.0;
    let gamma_fn = g.sqrt() * (2.0 / (g + 1.0)).powf((g + 1.0) / (2.0 * (g - 1.0)));
    let mdot_ideal = gamma_fn * p0 * a_t / (gas.r_specific_si * t0).sqrt();
    let m_e = physics::mach_from_area_ratio(8.0, g, true);
    let t_e = t0 / (1.0 + 0.5 * (g - 1.0) * m_e * m_e);
    let p_e = p0 * (t_e / t0).powf(g / (g - 1.0));
    let u_e = m_e * (g * gas.r_specific_si * t_e).sqrt();
    let thrust_ideal = mdot_ideal * u_e + (p_e - p_a * p0) * a_e;
    let cf_ideal = thrust_ideal / (p0 * a_t);

    let mdot_ratio = r.mass_flow_kg_s / mdot_ideal;
    let mach_err = (r.exit_mach - m_e).abs() / m_e;
    let cf_ratio = r.thrust_coefficient / cf_ideal;
    println!("T8: converged {} in {steps} steps (residual {:.2e})", r.converged, info.residual);
    println!("T8: mdot {:.3} kg/s / ideal {:.3} = {:.4} (pass 0.94-1.00)",
             r.mass_flow_kg_s, mdot_ideal, mdot_ratio);
    println!("T8: exit Mach {:.4} vs ideal {:.4} ({:+.2}%, pass +/-4%)",
             r.exit_mach, m_e, 100.0 * (r.exit_mach - m_e) / m_e);
    println!("T8: C_f {:.4} / ideal {:.4} = {:.4} (pass 0.975-0.995)",
             r.thrust_coefficient, cf_ideal, cf_ratio);
    println!("T8: thrust {:.0} N, c* {:.1} m/s, C_d {:.4}, p_e/p_a {:.3}, N_throat {:.1}, \
              confidence {:?}", r.thrust_n, r.c_star_m_s, r.discharge_coefficient,
             r.exit_pressure_ratio, r.cells_per_throat_radius, r.confidence);
    assert!((0.94..=1.00).contains(&mdot_ratio), "mdot/ideal = {mdot_ratio:.4}");
    assert!(mach_err <= 0.04, "exit Mach off ideal by {:.2}%", 100.0 * mach_err);
    assert!((0.975..=0.995).contains(&cf_ratio), "C_f/ideal = {cf_ratio:.4}");
}
