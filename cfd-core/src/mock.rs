//! `MockSolver` — an analytic expanding-cone plume behind the same `Solver`
//! trait as the real thing. Quasi-1D isentropic inside the nozzle, a
//! Prandtl-Meyer-style expansion at the lip, periodic shock cells, a Mach
//! disk, and a startup front that moves. Written by the coordinator; it is
//! what sessions C and D run against from minute one, and it is abort-ladder
//! rung 0 (`SolverKind::Mock` — watermark ANALYTIC PREVIEW, NOT A CFD
//! SOLUTION). Not owned by session A or B; it calls nothing in `kernel.rs`
//! or `physics.rs`.

use std::sync::Arc;

use cfd_contract::{
    Ambient, CfdError, Confidence, Numerics, Prim, Real, Report, Result, Snapshot,
    SolidField, SolveSetup, Solver, StepInfo,
};

use crate::step::snapshot_from_prims;

/// Fixed non-dimensional step; the mock has no CFL constraint.
const DT: f64 = 0.05;
/// Residual e-folding time; "settled" (1e-3) in roughly 7 time units.
const TAU: f64 = 1.0;

pub struct MockSolver {
    setup: SolveSetup,
    info: StepInfo,
    /// Bore radius per column: axis to the first solid cell, non-dimensional.
    /// f64::INFINITY where the column has no solid (open plume region).
    r_open: Vec<f64>,
    lip: Option<usize>,
    i_throat: usize,
    r_throat: f64,
    /// Time of the last ambient/geometry change; the residual restarts there.
    t_change: f64,
}

impl MockSolver {
    pub fn new(setup: SolveSetup) -> Result<Self> {
        if setup.solid.grid != setup.grid || setup.solid.fraction.len() != setup.grid.len() {
            return Err(CfdError::Geometry("solid field does not match grid".into()));
        }
        let mut s = MockSolver {
            info: StepInfo { step: 0, time: 0.0, dt: DT as Real, residual: f64::NAN,
                             converged: false, floor_activations: 0 },
            r_open: Vec::new(),
            lip: None,
            i_throat: 0,
            r_throat: 0.0,
            t_change: 0.0,
            setup,
        };
        s.recompute_profile();
        Ok(s)
    }

    fn recompute_profile(&mut self) {
        let g = self.setup.grid.clone();
        let solid = &self.setup.solid;
        self.r_open = (0..g.nz).map(|iz| {
            (0..g.nr).find(|&ir| solid.is_solid(g.idx(iz, ir)))
                .map(|ir| (g.r_face(ir) as f64).max(g.dr(0) as f64))
                .unwrap_or(f64::INFINITY)
        }).collect();
        self.lip = (0..g.nz).filter(|&iz| self.r_open[iz].is_finite()).max();
        if let Some(lip) = self.lip {
            let (i, r) = self.r_open[..=lip].iter().cloned().enumerate()
                .fold((0usize, f64::INFINITY), |acc, (i, r)| if r < acc.1 { (i, r) } else { acc });
            self.i_throat = i;
            self.r_throat = r.max(1e-6);
        }
    }

    /// Isentropic state at Mach m, chamber-referenced (p0 = t0 = 1):
    /// returns (rho, p, a).
    fn isentropic(&self, m: f64) -> (f64, f64, f64) {
        let g = self.setup.gas.gamma as f64;
        let t = 1.0 / (1.0 + 0.5 * (g - 1.0) * m * m);
        let p = t.powf(g / (g - 1.0));
        (p / t, p, (g * t).sqrt())
    }

    fn ambient_prim(&self) -> Prim {
        let a = self.setup.ambient;
        [a.p / a.t, 0.0, 0.0, a.p]
    }

    /// The analytic field: canonical primitives at cell (iz, ir).
    fn prim_at(&self, iz: usize, ir: usize) -> Prim {
        let g = self.setup.grid.clone();
        let gam = self.setup.gas.gamma as f64;
        let idx = g.idx(iz, ir);
        if self.setup.solid.is_solid(idx) { return [0.0; 4]; }
        let ambient = self.ambient_prim();
        let Some(lip) = self.lip else { return ambient; };

        let z = g.z_center(iz) as f64;
        let r = g.r_center(ir) as f64;
        let z_lip = g.z_face(lip + 1) as f64;
        let r_e = self.r_open[lip].min(g.lr()).max(self.r_throat);
        let p_a = (self.setup.ambient.p as f64).max(1e-9);

        // Exit conditions from the quasi-1D area ratio.
        let ar_e = (r_e / self.r_throat).powi(2).max(1.0);
        let m_e = mach_from_area_ratio_mock(ar_e, gam, true);
        let (_, p_e, a_e) = self.isentropic(m_e);
        let u_e = m_e * a_e;

        // Startup front leaves the lip at the exit velocity.
        let z_front = z_lip + u_e * self.info.time;

        if iz <= lip {
            // Inside the nozzle: quasi-1D isentropic, column by column.
            if !self.r_open[iz].is_finite() { return ambient; } // gap in the wall stroke
            let r_w = self.r_open[iz].max(self.r_throat);
            if r > r_w { return ambient; } // above the wall stroke
            let ar = (r_w / self.r_throat).powi(2).max(1.0);
            let m = mach_from_area_ratio_mock(ar, gam, iz > self.i_throat);
            let (rho, p, a) = self.isentropic(m);
            let u = m * a;
            // Radial velocity follows the local wall slope, linear in r.
            let iz2 = (iz + 1).min(lip);
            let r_w2 = if self.r_open[iz2].is_finite() { self.r_open[iz2] } else { r_w };
            let slope = (r_w2.max(self.r_throat) - r_w) / g.dz(iz) as f64;
            let v = u * slope * (r / r_w).clamp(0.0, 1.0);
            return [rho as Real, u as Real, v as Real, p as Real];
        }

        // Plume. Everything past the startup front is still ambient.
        if z > z_front { return ambient; }
        let s = z - z_lip;

        // Mach-disk station: sonic-orifice Ashkenas-Sherman scaled to the
        // exit, clamped inside the domain. Plausible, not physical.
        let npr = (p_e / p_a).clamp(0.05, 400.0);
        let z_m = (0.67 * (1.0 / p_a).sqrt() * 2.0 * r_e).clamp(2.0 * r_e, 0.9
            * (g.lz() - z_lip));
        let t_disk = (s / z_m).clamp(0.0, 1.0);

        // Barrel-shaped plume boundary; fatter when under-expanded.
        let fatness = 1.0 + 0.9 * (npr.powf(0.25) - 1.0).clamp(-0.5, 1.2);
        let r_b = r_e * (1.0 + (fatness - 1.0) * (4.0 * t_disk * (1.0 - t_disk)).sqrt());
        if r > r_b { return ambient; }

        let core = 1.0 - 0.4 * (r / r_b).powi(2);
        if s > z_m && r < 0.4 * r_e {
            // Behind the Mach disk: hot, slow, subsonic core.
            let m = 0.35 * core;
            let (rho, p_i, a) = self.isentropic(m);
            let p = p_a.max(p_i);
            return [(rho * p / p_i) as Real, (m * a) as Real, 0.0, p as Real];
        }

        // Expanding supersonic plume with breathing shock cells.
        let m_c = m_e + (2.2 * npr.powf(0.15) - 1.0).max(0.2) * t_disk;
        let m = (m_c * core).max(0.2);
        let (rho, p_i, a) = self.isentropic(m);
        let l_cell = 1.3 * r_e * (m_e * m_e - 1.0).max(0.1).sqrt();
        let phase = 0.4 * (0.6 * self.info.time).sin();
        let cells = 1.0 + 0.28 * (-1.5 * t_disk).exp()
            * (2.0 * std::f64::consts::PI * s / l_cell + phase).sin();
        // Pressure relaxes from the exit value toward ambient along the plume.
        let p = (p_e * (1.0 - t_disk) + p_a * t_disk) * cells;
        let u = m * a;
        // Radial spreading grows toward the boundary, stronger under-expanded.
        let v = u * (r / r_b) * 0.12 * (npr.powf(0.2)).clamp(0.5, 2.5)
            * (1.0 - t_disk);
        [(rho * p / p_i) as Real, u as Real, v as Real, p as Real]
    }
}

impl Solver for MockSolver {
    fn step(&mut self) -> Result<StepInfo> {
        self.info.step += 1;
        self.info.time += DT;
        self.info.dt = DT as Real;
        let settled = (self.info.time - self.t_change) / TAU;
        self.info.residual = if self.info.step >= 10 { (-settled).exp().max(1e-4) }
                             else { f64::NAN };
        self.info.converged = self.info.step >= 10 && self.info.residual < 1e-3;
        Ok(self.info)
    }

    fn run(&mut self, max_steps: u64) -> Result<StepInfo> {
        let mut info = self.info;
        for _ in 0..max_steps {
            info = self.step()?;
            if info.converged { break; }
        }
        Ok(info)
    }

    fn snapshot(&self) -> Snapshot {
        let g = self.setup.grid.clone();
        let prims: Vec<Prim> = (0..g.nr)
            .flat_map(|ir| (0..g.nz).map(move |iz| (iz, ir)))
            .map(|(iz, ir)| self.prim_at(iz, ir))
            .collect();
        snapshot_from_prims(&self.setup, &prims, self.info)
    }

    fn report(&self) -> Report {
        // Pure quasi-1D ideal numbers — this is an analytic preview, C_d = 1.
        let g = self.setup.gas.gamma as f64;
        let refs = self.setup.refs;
        let p_a = self.setup.ambient.p as f64;
        let empty = Report {
            mass_flow_kg_s: f64::NAN, thrust_n: f64::NAN, thrust_coefficient: f64::NAN,
            c_star_m_s: f64::NAN, discharge_coefficient: f64::NAN, exit_mach: f64::NAN,
            exit_pressure_pa: f64::NAN, exit_pressure_ratio: f64::NAN,
            ideal_exit_mach: f64::NAN, cells_per_throat_radius: f64::NAN,
            converged: self.info.converged, confidence: Confidence::NotConverged,
        };
        let Some(lip) = self.lip else { return empty; };
        let g_grid = &self.setup.grid;
        let r_t = self.r_throat;
        let r_e = self.r_open[lip].min(g_grid.lr()).max(r_t);
        let ar = (r_e / r_t).powi(2).max(1.0);
        let m_e = mach_from_area_ratio_mock(ar, g, true);
        let (_, p_e, a_e) = self.isentropic(m_e);
        let u_e = m_e * a_e;
        let a_t = std::f64::consts::PI * r_t * r_t;
        let a_e_area = std::f64::consts::PI * r_e * r_e;
        // Vandenkerckhove mass flow, non-dimensional (p0 = T0 = R = 1).
        let gamma_fn = g.sqrt() * (2.0 / (g + 1.0)).powf((g + 1.0) / (2.0 * (g - 1.0)));
        let mdot = gamma_fn * a_t;
        let thrust = mdot * u_e + (p_e - p_a) * a_e_area;
        let mdot_si = mdot * refs.rho_kg_m3 * refs.u_m_s * refs.l_m * refs.l_m;
        let force_scale = refs.p_pa * refs.l_m * refs.l_m;
        let a_t_si = a_t * refs.l_m * refs.l_m;
        let sep = (1.88 * m_e - 1.0).powf(-0.64).max(0.40);
        let pr = p_e / p_a;
        Report {
            mass_flow_kg_s: mdot_si,
            thrust_n: thrust * force_scale,
            thrust_coefficient: thrust / a_t,
            c_star_m_s: refs.p_pa * a_t_si / mdot_si,
            discharge_coefficient: 1.0,
            exit_mach: m_e,
            exit_pressure_pa: p_e * refs.p_pa,
            exit_pressure_ratio: pr,
            ideal_exit_mach: m_e,
            cells_per_throat_radius: r_t / self.setup.grid.dr_min() as f64,
            converged: self.info.converged,
            confidence: if !self.info.converged { Confidence::NotConverged }
                        else if pr < sep { Confidence::SeparationLikely }
                        else { Confidence::Valid },
        }
    }

    fn set_ambient(&mut self, a: Ambient) {
        self.setup.ambient = a;
        self.t_change = self.info.time;
        self.info.converged = false;
    }

    fn set_geometry(&mut self, solid: Arc<SolidField>) -> Result<()> {
        if solid.grid != self.setup.grid || solid.fraction.len() != self.setup.grid.len() {
            return Err(CfdError::Geometry("solid field does not match grid".into()));
        }
        self.setup.solid = solid;
        self.recompute_profile();
        self.t_change = self.info.time;
        self.info.converged = false;
        Ok(())
    }

    fn set_numerics(&mut self, n: Numerics) { self.setup.numerics = n; }

    fn step_count(&self) -> u64 { self.info.step }
}

/// Mock-only area-Mach inversion by bisection. The REAL one — NASA b4wind
/// Newton, in `physics.rs` — is session B's; this private copy exists so the
/// mock works before B lands and never has to call B's file.
fn mach_from_area_ratio_mock(ar: f64, gamma: f64, supersonic: bool) -> f64 {
    if (ar - 1.0).abs() < 1e-9 { return 1.0; }
    let area = |m: f64| -> f64 {
        let e = (gamma + 1.0) / (2.0 * (gamma - 1.0));
        (1.0 / m) * ((2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * m * m)).powf(e)
    };
    let (mut lo, mut hi) = if supersonic { (1.0, 100.0) } else { (1e-6, 1.0) };
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        // A/At decreases with M below sonic and increases above.
        let above = area(mid) > ar;
        if above == supersonic { hi = mid; } else { lo = mid; }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfd_contract::{Chamber, GasModel, Grid, RefScales};

    fn demo_setup() -> SolveSetup {
        // The demo case: gamma 1.24, R 378, p0 5 MPa, T0 3200 K, r_t 50 mm.
        let grid = Grid::uniform(80, 50, 0.29, 0.1);
        let gas = GasModel { gamma: 1.24, r_specific_si: 378.0 };
        // A crude converging-diverging wall: solid above a V-shaped contour.
        let mut solid = SolidField::empty(grid.clone());
        let lip = 30usize;
        for iz in 0..=lip {
            let r_wall = if iz < 10 { 2.0 } else if iz < 15 { 2.0 - 0.2 * (iz - 10) as f32 }
                         else { 1.0 + 0.12 * (iz - 15) as f32 };
            for ir in 0..grid.nr {
                if grid.r_center(ir) > r_wall && grid.r_center(ir) < 4.0 {
                    solid.fraction[grid.idx(iz, ir)] = 1.0;
                }
            }
        }
        SolveSetup {
            grid,
            solid: Arc::new(solid),
            gas,
            chamber: Chamber { p0: 1.0, t0: 1.0 },
            ambient: Ambient { p: 101325.0 / 5.0e6, t: 288.15 / 3200.0 },
            numerics: Numerics::default(),
            refs: RefScales::from_chamber(0.05, 5.0e6, 3200.0, &gas),
        }
    }

    #[test]
    fn mock_produces_a_moving_supersonic_plume() {
        let mut m = MockSolver::new(demo_setup()).unwrap();
        m.step().unwrap();
        let early = m.snapshot();
        for _ in 0..200 { m.step().unwrap(); }
        let late = m.snapshot();
        // The startup front moved: the late plume reaches further downstream.
        let mach_late = late.field(cfd_contract::FieldKind::Mach);
        let mach_early = early.field(cfd_contract::FieldKind::Mach);
        let far = late.grid.idx(70, 0);
        assert!(mach_late[far] > 1.0, "no supersonic flow far downstream at t=10");
        assert!(mach_early[far] < mach_late[far], "plume did not advance");
        // Supersonic at the nozzle exit, subsonic in the chamber.
        assert!(late.sample(cfd_contract::FieldKind::Mach, 30, 0) > 1.5);
        assert!(late.sample(cfd_contract::FieldKind::Mach, 2, 0) < 0.5);
        // SI sanity: chamber-adjacent pressure below but near p0.
        let p = late.sample(cfd_contract::FieldKind::Pressure, 2, 0);
        assert!(p > 1.0e6 && p < 5.0e6, "chamber pressure {p} Pa implausible");
        // Ranges come from fluid cells only and are finite.
        for k in cfd_contract::FieldKind::ALL {
            let (lo, hi) = late.range(k);
            assert!(lo.is_finite() && hi.is_finite() && hi >= lo);
        }
    }

    #[test]
    fn mock_report_matches_quasi_1d_identities() {
        let mut m = MockSolver::new(demo_setup()).unwrap();
        for _ in 0..200 { m.step().unwrap(); }
        let r = m.report();
        assert!(r.converged);
        // c* x mdot = p0 x A_t by construction.
        let a_t_si = std::f64::consts::PI * r.cells_per_throat_radius
            * m.setup.grid.dr(0) as f64 * m.setup.refs.l_m
            * (r.cells_per_throat_radius * m.setup.grid.dr(0) as f64 * m.setup.refs.l_m);
        assert!((r.c_star_m_s * r.mass_flow_kg_s / (5.0e6 * a_t_si) - 1.0).abs() < 1e-9);
        assert_eq!(r.discharge_coefficient, 1.0);
        assert!(r.exit_mach > 1.0 && r.ideal_exit_mach == r.exit_mach);
        assert!(r.thrust_coefficient > 0.5 && r.thrust_coefficient < 2.5);
    }

    #[test]
    fn mock_area_mach_round_trips() {
        for gamma in [1.2, 1.24, 1.4] {
            let e = (gamma + 1.0) / (2.0 * (gamma - 1.0));
            let area = |m: f64| (1.0 / m)
                * ((2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * m * m)).powf(e);
            for m in [0.05, 0.3, 0.9, 1.5, 3.0, 8.0] {
                let got = mach_from_area_ratio_mock(area(m), gamma, m > 1.0);
                assert!((got - m).abs() < 1e-6, "gamma {gamma} M {m} -> {got}");
            }
        }
        assert_eq!(mach_from_area_ratio_mock(1.0, 1.4, true), 1.0);
    }
}
