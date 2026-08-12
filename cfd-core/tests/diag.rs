//! Integration diagnostics — field profiles behind the T7/T8 ladder failures.
//! Not part of the ladder; run by hand:
//!     cargo test -p cfd-core --test diag -- --include-ignored --nocapture

use std::sync::Arc;

use cfd_contract::{
    Ambient, Chamber, FieldKind, GasModel, Geometry, Grid, Numerics, RefScales, SolidField,
    SolveSetup, Solver,
};
use cfd_core::EulerSolver;

fn identity_refs() -> RefScales {
    RefScales { l_m: 1.0, p_pa: 1.0, rho_kg_m3: 1.0, u_m_s: 1.0, t_k: 1.0, time_s: 1.0 }
}

/// T7 cone flow: radial Mach and pressure profiles at z = 2.0, plus the Mach
/// value cell-by-cell above the surface, to size the wall entropy layer.
#[test]
#[ignore = "diagnostic"]
fn diag_cone_profiles() {
    let gamma = 1.4f64;
    let m_inf = 2.35f64;
    let grid = Grid { nz: 300, nr: 220, dz: 0.01, dr: 0.01 };
    let z0 = 0.5f64;
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
    let t0 = 1.0 + 0.5 * (gamma - 1.0) * m_inf * m_inf;
    let setup = SolveSetup {
        grid,
        solid: Arc::new(solid),
        gas: GasModel { gamma: gamma as f32, r_specific_si: 287.0 },
        chamber: Chamber { p0: t0.powf(3.5) as f32, t0: t0 as f32 },
        ambient: Ambient { p: 1.0, t: 1.0 },
        numerics: Numerics { geometry: Geometry::Axisymmetric, quasi1d_init: false,
                             sponge_cells: 0, ..Numerics::default() },
        refs: identity_refs(),
    };
    let mut s = EulerSolver::new(setup).unwrap();
    s.set_initial(|_, _| [1.0, u_inf as f32, 0.0, 1.0]);
    let mut info = s.step().unwrap();
    while info.time < 4.0 { info = s.step().unwrap(); }
    let snap = s.snapshot();
    let iz = (2.0 / grid.dz as f64) as usize; // z = 2.0, surface at r = 0.264
    println!("cone radial profile at z=2.0 (surface r=0.264, shock ~r=0.75):");
    println!("  ir     r     solid   M       p       uz      ur");
    for ir in 20..45 {
        println!("  {ir:3}  {:.3}  {}  {:.4}  {:.4}  {:+.4}  {:+.4}",
                 grid.r_center(ir),
                 snap.solid.is_solid(grid.idx(iz, ir)) as u8,
                 snap.sample(FieldKind::Mach, iz, ir),
                 snap.sample(FieldKind::Pressure, iz, ir),
                 snap.sample(FieldKind::VelocityZ, iz, ir),
                 snap.sample(FieldKind::VelocityR, iz, ir));
    }
    println!("first-fluid-above-surface M along z:");
    for iz in (60..260).step_by(20) {
        let z = (iz as f64 + 0.5) * grid.dz as f64;
        if z <= z0 { continue; }
        let mut ir = 0;
        while snap.solid.is_solid(grid.idx(iz, ir)) { ir += 1; }
        println!("  z {:.2}: surf ir {}  M[0] {:.3}  M[+1] {:.3}  M[+2] {:.3}  M[+4] {:.3}",
                 z, ir,
                 snap.sample(FieldKind::Mach, iz, ir),
                 snap.sample(FieldKind::Mach, iz, ir + 1),
                 snap.sample(FieldKind::Mach, iz, ir + 2),
                 snap.sample(FieldKind::Mach, iz, ir + 4));
    }
}

/// T8 nozzle: centerline Mach vs z at several step counts, exit radial
/// profile, and the report — is the core expansion wrong or only the wall
/// band, and is 12k steps simply not steady yet?
#[test]
#[ignore = "diagnostic"]
fn diag_nozzle_centerline() {
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
    // Lip column and bore for orientation.
    let lip = (0..grid.nz)
        .filter(|&iz| (0..grid.nr).any(|ir| solid.is_solid(grid.idx(iz, ir))))
        .max()
        .unwrap();
    let setup = SolveSetup {
        grid,
        solid: Arc::new(solid),
        gas,
        chamber: Chamber { p0: 1.0, t0: 1.0 },
        ambient: Ambient { p: (101_325.0 / refs.p_pa) as f32, t: (288.15 / refs.t_k) as f32 },
        numerics: Numerics::default(),
        refs,
    };
    let mut s = EulerSolver::new(setup).unwrap();
    let centerline = |s: &EulerSolver, label: &str| {
        let snap = s.snapshot();
        let m: Vec<String> = (0..=lip + 40)
            .step_by(8)
            .map(|iz| format!("{:.2}", snap.sample(FieldKind::Mach, iz, 0)))
            .collect();
        println!("{label}: centerline M every 8 cols to lip+40 (lip = {lip}): {}", m.join(" "));
    };
    centerline(&s, "step 0 (quasi-1D init)");
    let mut info = s.step().unwrap();
    for target in [1000u64, 3000, 6000, 12000, 20000] {
        while info.step < target {
            info = s.step().unwrap();
        }
        centerline(&s, &format!("step {target} (residual {:.2e})", info.residual));
        let r = s.report();
        println!("  report: mdot {:.3} kg/s, exit M {:.3}, C_f {:.4}, p_e/p_a {:.3}, floors {}",
                 r.mass_flow_kg_s, r.exit_mach, r.thrust_coefficient, r.exit_pressure_ratio,
                 info.floor_activations);
    }
    // Exit radial Mach profile.
    let snap = s.snapshot();
    println!("exit column (lip = {lip}) radial Mach / p, every 4th row:");
    for ir in (0..64).step_by(4) {
        println!("  r {:.2}: M {:.3}  p/p_a {:.3}  solid {}",
                 grid.r_center(ir),
                 snap.sample(FieldKind::Mach, lip, ir),
                 snap.sample(FieldKind::Pressure, lip, ir) as f64 / 101_325.0,
                 snap.solid.is_solid(grid.idx(lip, ir)) as u8);
    }
}

/// Does the carbuncle sensor (physics-reference §2) fire on the SMOOTH steep
/// expansion in the divergent section? The Omega = min/max p test cannot tell
/// a shock from an expansion with the same pressure ratio across its window,
/// and masked cells get dissipative HLL radial fluxes.
#[test]
#[ignore = "diagnostic"]
fn diag_carbuncle_mask_in_nozzle() {
    use cfd_contract::kernels::cons_to_prim;
    use cfd_contract::{Prim, NG};
    let gas = GasModel { gamma: 1.24, r_specific_si: 378.0 };
    let refs = RefScales::from_chamber(0.05, 5.0e6, 3200.0, &gas);
    let grid = Grid { nz: 320, nr: 200, dz: 0.1449, dr: 0.05 };
    let spec = cfd_geom::NozzleSpec {
        throat_radius_m: 0.05, area_ratio: 8.0, contraction_ratio: 4.0,
        converge_half_angle_deg: 30.0, throat_arc_up: 1.5, throat_arc_down: 0.382,
        contour: cfd_geom::ContourKind::Conical { half_angle_deg: 15.0 },
    };
    let wall = cfd_geom::generate_contour(&spec, 512).unwrap();
    let solid = cfd_geom::rasterize(&wall, &grid, &refs).unwrap();
    let setup = SolveSetup {
        grid, solid: Arc::new(solid), gas,
        chamber: Chamber { p0: 1.0, t0: 1.0 },
        ambient: Ambient { p: (101_325.0 / refs.p_pa) as f32, t: (288.15 / refs.t_k) as f32 },
        numerics: Numerics::default(),
        refs,
    };
    // Quasi-1D init state — perfectly smooth, shock-free by construction.
    let s = EulerSolver::new(setup).unwrap();
    let snap = s.snapshot();
    let mut w: Vec<Prim> = vec![[1.0, 0.0, 0.0, 1.0]; grid.glen()];
    let ng = NG as isize;
    for ir in -ng..(grid.nr as isize + ng) {
        for iz in -ng..(grid.nz as isize + ng) {
            let cz = iz.clamp(0, grid.nz as isize - 1) as usize;
            let cr = ir.clamp(0, grid.nr as isize - 1) as usize;
            let i = grid.idx(cz, cr);
            let (rho, p) = (snap.sample(FieldKind::Density, cz, cr) / refs.rho_kg_m3 as f32,
                            snap.sample(FieldKind::Pressure, cz, cr) / refs.p_pa as f32);
            let (uz, ur) = (snap.sample(FieldKind::VelocityZ, cz, cr) / refs.u_m_s as f32,
                            snap.sample(FieldKind::VelocityR, cz, cr) / refs.u_m_s as f32);
            w[grid.gidx(iz, ir)] = if snap.solid.is_solid(i) || rho <= 0.0 {
                [1.0, 0.0, 0.0, 1.0]
            } else {
                let _ = cons_to_prim; // (prims assembled directly from SI fields)
                [rho, uz, ur, p]
            };
        }
    }
    let mut mask = vec![false; grid.glen()];
    cfd_core::physics::carbuncle_mask(&w, &grid, &mut mask);
    let mut per_col = vec![0usize; grid.nz];
    for ir in 0..grid.nr {
        for iz in 0..grid.nz {
            if mask[grid.gidx(iz as isize, ir as isize)]
                && !snap.solid.is_solid(grid.idx(iz, ir)) {
                per_col[iz] += 1;
            }
        }
    }
    let total: usize = per_col.iter().sum();
    println!("carbuncle mask on the SMOOTH quasi-1D init: {total} fluid cells masked");
    println!("masked cells per column, columns 20..90 (throat ~30, lip 75):");
    for iz in (20..90).step_by(5) {
        println!("  col {iz:3} (z {:5.2}): {:3} masked", (iz as f32 + 0.5) * grid.dz, per_col[iz]);
    }
}

/// T8 on the measure grid (640x400, N_throat = 40): does grid refinement move
/// the nozzle numbers toward the pass bands, as physics-reference §8 predicts
/// (mdot error halves; the 2.6-axial-cell throat arc was the weakest link)?
#[test]
#[ignore = "diagnostic"]
fn diag_nozzle_measure_grid() {
    let gas = GasModel { gamma: 1.24, r_specific_si: 378.0 };
    let refs = RefScales::from_chamber(0.05, 5.0e6, 3200.0, &gas);
    let grid = Grid { nz: 640, nr: 400, dz: 0.0724, dr: 0.025 };
    let spec = cfd_geom::NozzleSpec {
        throat_radius_m: 0.05, area_ratio: 8.0, contraction_ratio: 4.0,
        converge_half_angle_deg: 30.0, throat_arc_up: 1.5, throat_arc_down: 0.382,
        contour: cfd_geom::ContourKind::Conical { half_angle_deg: 15.0 },
    };
    let wall = cfd_geom::generate_contour(&spec, 1024).unwrap();
    let solid = cfd_geom::rasterize(&wall, &grid, &refs).unwrap();
    let setup = SolveSetup {
        grid, solid: Arc::new(solid), gas,
        chamber: Chamber { p0: 1.0, t0: 1.0 },
        ambient: Ambient { p: (101_325.0 / refs.p_pa) as f32, t: (288.15 / refs.t_k) as f32 },
        numerics: Numerics::default(),
        refs,
    };
    let mut s = EulerSolver::new(setup).unwrap();
    let mut info = s.step().unwrap();
    for target in [8000u64, 16000] {
        while info.step < target { info = s.step().unwrap(); }
        let r = s.report();
        let mdot_ideal = 23.4297; // Vandenkerckhove, gamma 1.24, this chamber
        println!("measure grid step {target}: mdot {:.3} kg/s (/ideal {:.4}), exit M {:.3}, \
                  C_f {:.4}, p_e/p_a {:.3}, floors {}, residual {:.2e}",
                 r.mass_flow_kg_s, r.mass_flow_kg_s / mdot_ideal, r.exit_mach,
                 r.thrust_coefficient, r.exit_pressure_ratio, info.floor_activations,
                 info.residual);
    }
}
