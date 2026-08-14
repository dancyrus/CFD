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
    let grid = Grid::uniform(300, 220, 0.01, 0.01);
    let z0 = 0.5f64;
    let tan_c = 10.0f64.to_radians().tan();
    let mut solid = SolidField::empty(grid.clone());
    for iz in 0..grid.nz {
        let z = (iz as f64 + 0.5) * grid.dz(0) as f64;
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
        grid: grid.clone(),
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
    let iz = (2.0 / grid.dz(0) as f64) as usize; // z = 2.0, surface at r = 0.264
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
        let z = (iz as f64 + 0.5) * grid.dz(0) as f64;
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
    let grid = Grid::uniform(320, 200, 0.1449, 0.05);
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
        grid: grid.clone(),
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
    let grid = Grid::uniform(320, 200, 0.1449, 0.05);
    let spec = cfd_geom::NozzleSpec {
        throat_radius_m: 0.05, area_ratio: 8.0, contraction_ratio: 4.0,
        converge_half_angle_deg: 30.0, throat_arc_up: 1.5, throat_arc_down: 0.382,
        contour: cfd_geom::ContourKind::Conical { half_angle_deg: 15.0 },
    };
    let wall = cfd_geom::generate_contour(&spec, 512).unwrap();
    let solid = cfd_geom::rasterize(&wall, &grid, &refs).unwrap();
    let setup = SolveSetup {
        grid: grid.clone(), solid: Arc::new(solid), gas,
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
        println!("  col {iz:3} (z {:5.2}): {:3} masked", (iz as f32 + 0.5) * grid.dz(0), per_col[iz]);
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
    let grid = Grid::uniform(640, 400, 0.0724, 0.025);
    let spec = cfd_geom::NozzleSpec {
        throat_radius_m: 0.05, area_ratio: 8.0, contraction_ratio: 4.0,
        converge_half_angle_deg: 30.0, throat_arc_up: 1.5, throat_arc_down: 0.382,
        contour: cfd_geom::ContourKind::Conical { half_angle_deg: 15.0 },
    };
    let wall = cfd_geom::generate_contour(&spec, 1024).unwrap();
    let solid = cfd_geom::rasterize(&wall, &grid, &refs).unwrap();
    let setup = SolveSetup {
        grid: grid.clone(), solid: Arc::new(solid), gas,
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

/// T8 failure diagnosis, in the order asked: (1) convergence history with the
/// report values every 2000 steps, (2) fine centerline Mach from throat to
/// exit plus the ambient-vs-design condition, (3) the exact exit-plane
/// sampling: which column, which cells, what solid fractions sit there, and
/// how the integrals move if sampled one or two columns upstream.
#[test]
#[ignore = "diagnostic"]
fn diag_t8_convergence_and_exit_plane() {
    let gas = GasModel { gamma: 1.24, r_specific_si: 378.0 };
    let refs = RefScales::from_chamber(0.05, 5.0e6, 3200.0, &gas);
    let grid = Grid::uniform(320, 200, 0.1449, 0.05);
    let spec = cfd_geom::NozzleSpec {
        throat_radius_m: 0.05, area_ratio: 8.0, contraction_ratio: 4.0,
        converge_half_angle_deg: 30.0, throat_arc_up: 1.5, throat_arc_down: 0.382,
        contour: cfd_geom::ContourKind::Conical { half_angle_deg: 15.0 },
    };
    let wall = cfd_geom::generate_contour(&spec, 512).unwrap();
    let solid = cfd_geom::rasterize(&wall, &grid, &refs).unwrap();

    // --- (3a) geometry bookkeeping before any stepping ---------------------
    let lip = (0..grid.nz)
        .filter(|&iz| (0..grid.nr).any(|ir| solid.is_solid(grid.idx(iz, ir))))
        .max().unwrap();
    let z_exit_contour = wall.points.last().unwrap()[0] / refs.l_m; // nondim
    println!("exit plane: lip column {lip}, cell z-range [{:.3}, {:.3}] r_t; \
              contour ends at z = {:.3} r_t",
             lip as f32 * grid.dz(0), (lip as f32 + 1.0) * grid.dz(0), z_exit_contour);
    for iz in [lip - 2, lip - 1, lip] {
        let first_solid = (0..grid.nr).find(|&ir| solid.is_solid(grid.idx(iz, ir))).unwrap();
        println!("  col {iz}: first solid row {first_solid} (r_face {:.3}); fractions \
                  rows {}..{}: {:?}",
                 grid.r_face(first_solid), first_solid.saturating_sub(2), first_solid + 1,
                 (first_solid.saturating_sub(2)..=first_solid + 1)
                     .map(|ir| solid.fraction[grid.idx(iz, ir)])
                     .collect::<Vec<_>>());
    }
    // Throat: minimum bore over nozzle columns.
    let (throat_col, throat_rows) = (0..=lip)
        .filter_map(|iz| (0..grid.nr).find(|&ir| solid.is_solid(grid.idx(iz, ir)))
            .map(|ir| (iz, ir)))
        .min_by_key(|&(_, ir)| ir).unwrap();
    println!("throat: column {throat_col}, first solid row {throat_rows} \
              (bore {:.3} r_t, N_throat = {})", grid.r_face(throat_rows), throat_rows);

    // --- ambient vs design -------------------------------------------------
    let p_a_nd = 101_325.0 / refs.p_pa;
    let g = gas.gamma as f64;
    let m_e = cfd_core::physics::mach_from_area_ratio(8.0, g, true);
    let p_e_ideal = (1.0 + 0.5 * (g - 1.0) * m_e * m_e).powf(-g / (g - 1.0));
    println!("ambient: p_a = 101325 Pa = {:.5} p0; ideal p_e = {:.5} p0 ({:.1} kPa); \
              p_e/p_a(design) = {:.3} -> {} at sea level (separation threshold 0.40)",
             p_a_nd, p_e_ideal, p_e_ideal * refs.p_pa / 1e3, p_e_ideal / p_a_nd,
             if p_e_ideal / p_a_nd < 1.0 { "OVERexpanded" } else { "UNDERexpanded" });

    // --- (1) convergence history ------------------------------------------
    let setup = SolveSetup {
        grid: grid.clone(), solid: Arc::new(solid), gas,
        chamber: Chamber { p0: 1.0, t0: 1.0 },
        ambient: Ambient { p: p_a_nd as f32, t: (288.15 / refs.t_k) as f32 },
        numerics: Numerics::default(),
        refs,
    };
    let mut s = EulerSolver::new(setup).unwrap();
    let mut info = s.step().unwrap();
    println!("step | residual | exit M | mdot kg/s | C_f | p_e/p_a");
    for target in (2000u64..=20000).step_by(2000) {
        while info.step < target { info = s.step().unwrap(); }
        let r = s.report();
        println!("{:5} | {:.3e} | {:.3} | {:.3} | {:.4} | {:.3}",
                 target, info.residual, r.exit_mach, r.mass_flow_kg_s,
                 r.thrust_coefficient, r.exit_pressure_ratio);
    }

    // --- (2) fine centerline Mach, throat to exit -------------------------
    let snap = s.snapshot();
    println!("centerline M, every column from throat ({throat_col}) to lip ({lip}):");
    let vals: Vec<String> = (throat_col..=lip)
        .map(|iz| format!("{:.2}", snap.sample(FieldKind::Mach, iz, 0))).collect();
    println!("  {}", vals.join(" "));

    // --- (3b) the integral, column by column near the exit ----------------
    println!("exit-plane integrals vs sampling column (fluid cells 0..first-solid):");
    for iz in [lip - 4, lip - 2, lip - 1, lip] {
        let first_solid = (0..grid.nr).find(|&ir| snap.solid.is_solid(grid.idx(iz, ir))).unwrap();
        let (mut mdot, mut mach_a, mut area) = (0.0f64, 0.0f64, 0.0f64);
        for ir in 0..first_solid {
            let da = 2.0 * std::f64::consts::PI * grid.r_center(ir) as f64 * grid.dr(0) as f64;
            let rho = snap.sample(FieldKind::Density, iz, ir) as f64;
            let uz = snap.sample(FieldKind::VelocityZ, iz, ir) as f64;
            mdot += rho * uz * da * refs.l_m * refs.l_m;
            mach_a += snap.sample(FieldKind::Mach, iz, ir) as f64 * da;
            area += da;
        }
        println!("  col {iz} (bore {} rows): mdot {:.3} kg/s, area-avg M {:.3}",
                 first_solid, mdot, mach_a / area);
    }
    // Radial Mach at the lip, fine near the wall.
    let first_solid = (0..grid.nr).find(|&ir| snap.solid.is_solid(grid.idx(lip, ir))).unwrap();
    println!("lip column radial M, last 14 fluid rows to the wall:");
    for ir in first_solid.saturating_sub(14)..first_solid {
        println!("  r {:.2}: M {:.3}", grid.r_center(ir), snap.sample(FieldKind::Mach, lip, ir));
    }
}

/// Step-2 measurement (NOT adopted): the T8 nozzle under
/// WallMode::ColumnReflect. Compares against the Mirror staircase to
/// attribute how much of the exit wall layer is the staircase wall itself.
#[test]
#[ignore = "diagnostic"]
fn diag_t8_column_reflect() {
    let gas = GasModel { gamma: 1.24, r_specific_si: 378.0 };
    let refs = RefScales::from_chamber(0.05, 5.0e6, 3200.0, &gas);
    let grid = Grid::uniform(320, 200, 0.1449, 0.05);
    let spec = cfd_geom::NozzleSpec {
        throat_radius_m: 0.05, area_ratio: 8.0, contraction_ratio: 4.0,
        converge_half_angle_deg: 30.0, throat_arc_up: 1.5, throat_arc_down: 0.382,
        contour: cfd_geom::ContourKind::Conical { half_angle_deg: 15.0 },
    };
    let wall = cfd_geom::generate_contour(&spec, 512).unwrap();
    let solid = cfd_geom::rasterize(&wall, &grid, &refs).unwrap();
    let lip = (0..grid.nz)
        .filter(|&iz| (0..grid.nr).any(|ir| solid.is_solid(grid.idx(iz, ir))))
        .max().unwrap();
    let setup = SolveSetup {
        grid: grid.clone(), solid: Arc::new(solid), gas,
        chamber: Chamber { p0: 1.0, t0: 1.0 },
        ambient: Ambient { p: (101_325.0 / refs.p_pa) as f32, t: (288.15 / refs.t_k) as f32 },
        numerics: Numerics { wall_mode: cfd_contract::WallMode::ColumnReflect,
                             ..Numerics::default() },
        refs,
    };
    let mut s = EulerSolver::new(setup).unwrap();
    let mut info = s.step().unwrap();
    for target in [4000u64, 8000, 12000] {
        while info.step < target { info = s.step().unwrap(); }
        let r = s.report();
        println!("ColumnReflect step {target}: mdot {:.3} (/ideal {:.4}), exit M {:.3}, \
                  C_f {:.4} (/ideal {:.4}), p_e/p_a {:.3}, floors {}, residual {:.2e}",
                 r.mass_flow_kg_s, r.mass_flow_kg_s / 23.4297, r.exit_mach,
                 r.thrust_coefficient, r.thrust_coefficient / 1.5313,
                 r.exit_pressure_ratio, info.floor_activations, info.residual);
    }
    let snap = s.snapshot();
    let first_solid = (0..grid.nr).find(|&ir| snap.solid.is_solid(grid.idx(lip, ir))).unwrap();
    println!("lip radial M, last 14 fluid rows (Mirror baseline fell 2.95 -> 0.02 here):");
    for ir in first_solid.saturating_sub(14)..first_solid {
        println!("  r {:.2}: M {:.3}", grid.r_center(ir), snap.sample(FieldKind::Mach, lip, ir));
    }
    println!("centerline M every 4 cols, throat to lip:");
    let vals: Vec<String> = (27..=lip).step_by(4)
        .map(|iz| format!("{:.2}", snap.sample(FieldKind::Mach, iz, 0))).collect();
    println!("  {}", vals.join(" "));
}

/// Step-3 measurement: T8 with dz refined 0.1449 -> 0.10 (physics-reference
/// §8's "first lever": the throat downstream arc was spanned by only 2.6
/// axial cells). Same dr, same tolerancing; prints the T8 numbers and where
/// they sit against the unchanged pass bands.
#[test]
#[ignore = "diagnostic"]
fn diag_t8_dz_010() {
    let gas = GasModel { gamma: 1.24, r_specific_si: 378.0 };
    let refs = RefScales::from_chamber(0.05, 5.0e6, 3200.0, &gas);
    let grid = Grid::uniform(464, 200, 0.10, 0.05);
    let spec = cfd_geom::NozzleSpec {
        throat_radius_m: 0.05, area_ratio: 8.0, contraction_ratio: 4.0,
        converge_half_angle_deg: 30.0, throat_arc_up: 1.5, throat_arc_down: 0.382,
        contour: cfd_geom::ContourKind::Conical { half_angle_deg: 15.0 },
    };
    let wall = cfd_geom::generate_contour(&spec, 512).unwrap();
    let solid = cfd_geom::rasterize(&wall, &grid, &refs).unwrap();
    let lip = (0..grid.nz)
        .filter(|&iz| (0..grid.nr).any(|ir| solid.is_solid(grid.idx(iz, ir))))
        .max().unwrap();
    let setup = SolveSetup {
        grid: grid.clone(), solid: Arc::new(solid), gas,
        chamber: Chamber { p0: 1.0, t0: 1.0 },
        ambient: Ambient { p: (101_325.0 / refs.p_pa) as f32, t: (288.15 / refs.t_k) as f32 },
        numerics: Numerics::default(),
        refs,
    };
    let g = gas.gamma as f64;
    let m_e = cfd_core::physics::mach_from_area_ratio(8.0, g, true);
    let t_e = 1.0 / (1.0 + 0.5 * (g - 1.0) * m_e * m_e);
    let p_e = t_e.powf(g / (g - 1.0));
    let u_e = m_e * (g * gas.r_specific_si * t_e * refs.t_k).sqrt();
    let a_t = std::f64::consts::PI * refs.l_m * refs.l_m;
    let mdot_ideal = g.sqrt() * (2.0 / (g + 1.0)).powf((g + 1.0) / (2.0 * (g - 1.0)))
        * refs.p_pa * a_t / (gas.r_specific_si * refs.t_k).sqrt();
    let cf_ideal = (mdot_ideal * u_e + (p_e - 101_325.0 / refs.p_pa as f64) * refs.p_pa
        * a_t * 8.0) / (refs.p_pa * a_t);
    let mut s = EulerSolver::new(setup).unwrap();
    let mut info = s.step().unwrap();
    for target in [8000u64, 12000, 16000] {
        while info.step < target { info = s.step().unwrap(); }
        let r = s.report();
        println!("dz=0.10 step {target}: mdot/ideal {:.4} (band 0.94-1.00), exit M {:.3} \
                  vs {:.3} ({:+.2}%, band +/-4%), C_f/ideal {:.4} (band 0.975-0.995), \
                  p_e/p_a {:.3}, floors {}, residual {:.2e}",
                 r.mass_flow_kg_s / mdot_ideal, r.exit_mach, m_e,
                 100.0 * (r.exit_mach - m_e) / m_e, r.thrust_coefficient / cf_ideal,
                 r.exit_pressure_ratio, info.floor_activations, info.residual);
    }
    let snap = s.snapshot();
    let first_solid = (0..grid.nr).find(|&ir| snap.solid.is_solid(grid.idx(lip, ir))).unwrap();
    println!("lip radial M, last 14 fluid rows:");
    for ir in first_solid.saturating_sub(14)..first_solid {
        println!("  r {:.2}: M {:.3}", grid.r_center(ir), snap.sample(FieldKind::Mach, lip, ir));
    }
}

/// Settling trend for the cut-domain wall-layer measurement: delta50 and exit
/// M at several step counts and two measurement columns, both resolutions.
#[test]
#[ignore = "diagnostic"]
fn diag_wall_layer_settling() {
    for (n_throat, steps) in [(20usize, [1500u64, 3000, 6000, 12000]),
                              (40, [3000, 6000, 12000, 24000])] {
        for target in steps {
            let (d_lip, d_mid, m, mdot, res) = wall_layer_probe(n_throat, target);
            println!("N{n_throat} step {target:5}: delta50 lip-4 {d_lip:.3} r_t, \
                      mid-divergent {d_mid:.3} r_t, exit M {m:.3}, mdot/ideal {mdot:.4}, \
                      residual {res:.2e}");
        }
    }
}

fn wall_layer_probe(n_throat: usize, steps: u64) -> (f64, f64, f64, f64, f64) {
    let gas = GasModel { gamma: 1.24, r_specific_si: 378.0 };
    let refs = RefScales::from_chamber(0.05, 5.0e6, 3200.0, &gas);
    let dr = 1.0f32 / n_throat as f32;
    let dz = 2.898 * dr;
    let grid = Grid::uniform((12.6 / dz).ceil() as usize, (4.0 / dr) as usize, dz, dr);
    let spec = cfd_geom::NozzleSpec {
        throat_radius_m: 0.05, area_ratio: 8.0, contraction_ratio: 4.0,
        converge_half_angle_deg: 30.0, throat_arc_up: 1.5, throat_arc_down: 0.382,
        contour: cfd_geom::ContourKind::Conical { half_angle_deg: 15.0 },
    };
    let wall = cfd_geom::generate_contour(&spec, 512).unwrap();
    let solid = cfd_geom::rasterize(&wall, &grid, &refs).unwrap();
    let lip = (0..grid.nz)
        .filter(|&iz| (0..grid.nr).any(|ir| solid.is_solid(grid.idx(iz, ir))))
        .max().unwrap();
    let throat = (0..=lip)
        .filter_map(|iz| (0..grid.nr).find(|&ir| solid.is_solid(grid.idx(iz, ir)))
            .map(|b| (iz, b)))
        .min_by_key(|&(_, b)| b).unwrap().0;
    let setup = SolveSetup {
        grid: grid.clone(), solid: Arc::new(solid), gas,
        chamber: Chamber { p0: 1.0, t0: 1.0 },
        ambient: Ambient { p: (101_325.0 / refs.p_pa) as f32, t: (288.15 / refs.t_k) as f32 },
        numerics: Numerics { sponge_cells: 0, ..Numerics::default() },
        refs,
    };
    let mut s = EulerSolver::new(setup).unwrap();
    let mut info = s.step().unwrap();
    while info.step < steps { info = s.step().unwrap(); }
    let snap = s.snapshot();
    let delta50 = |iz: usize| -> f64 {
        let b = (0..grid.nr).find(|&ir| snap.solid.is_solid(grid.idx(iz, ir))).unwrap();
        let core: f64 = (0..b / 2).map(|ir| snap.sample(FieldKind::Mach, iz, ir) as f64)
            .sum::<f64>() / (b / 2) as f64;
        for ir in (0..b).rev() {
            let m = snap.sample(FieldKind::Mach, iz, ir) as f64;
            if m >= 0.5 * core {
                return grid.r_face(b) as f64 - grid.r_center(ir) as f64;
            }
        }
        f64::NAN
    };
    let iz = lip - 4;
    let b = (0..grid.nr).find(|&ir| snap.solid.is_solid(grid.idx(iz, ir))).unwrap();
    let (mut mdot, mut mach_a, mut area) = (0.0f64, 0.0f64, 0.0f64);
    for ir in 0..b {
        let da = 2.0 * std::f64::consts::PI * grid.r_center(ir) as f64 * grid.dr(0) as f64;
        mdot += snap.sample(FieldKind::Density, iz, ir) as f64
            * snap.sample(FieldKind::VelocityZ, iz, ir) as f64 * da * refs.l_m * refs.l_m;
        mach_a += snap.sample(FieldKind::Mach, iz, ir) as f64 * da;
        area += da;
    }
    let g = gas.gamma as f64;
    let a_t = std::f64::consts::PI * refs.l_m * refs.l_m;
    let mdot_ideal = g.sqrt() * (2.0 / (g + 1.0)).powf((g + 1.0) / (2.0 * (g - 1.0)))
        * refs.p_pa * a_t / (gas.r_specific_si * refs.t_k).sqrt();
    (delta50(lip - 4), delta50((throat + lip) / 2), mach_a / area,
     mdot / mdot_ideal, info.residual)
}
