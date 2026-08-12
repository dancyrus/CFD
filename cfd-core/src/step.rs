//! SSP-RK2 orchestration and the `Solver` impl. **Frozen — coordinator wrote
//! it. Sessions A and B never open this file.** It calls the signatures
//! declared in `kernel.rs` (session A) and `physics.rs` (session B).
//!
//! Buffer ownership: this file allocates every solver-internal buffer. All of
//! them are padded (`glen()` long, `Grid::gidx` indexing): the ping-pong pair
//! in `State`, the stage buffer `u1`, primitives `w`, `rhs`, the carbuncle
//! `mask`, and `solid`, which is the thresholded solid field edge-replicated
//! into the ghost band.

use std::sync::Arc;

use cfd_contract::kernels::{cons_to_prim, prim_to_cons, sound_speed};
use cfd_contract::{
    Ambient, CfdError, Cons, FieldKind, Numerics, Prim, Real, Report, Result, Snapshot,
    SolidField, SolveSetup, Solver, State, StepInfo, Confidence, NG,
};

use crate::{kernel, physics};

/// Residual threshold for the green "settled" dot. docs/physics-reference.md §9.
const RESIDUAL_CONVERGED: f64 = 1e-3;

pub struct EulerSolver {
    setup: SolveSetup,
    state: State,
    u1: Vec<Cons>,
    w: Vec<Prim>,
    rhs: Vec<Cons>,
    mask: Vec<bool>,
    solid: Vec<bool>,
    info: StepInfo,
    ledger: physics::FlipLedger,
    residual_ref: f64,
    pending_geometry: Option<Arc<SolidField>>,
}

impl EulerSolver {
    pub fn new(setup: SolveSetup) -> Result<Self> {
        let g = setup.grid;
        if g.nz < 4 || g.nr < 1 {
            return Err(CfdError::Grid(format!("grid too small: {}x{}", g.nz, g.nr)));
        }
        if setup.solid.grid != g || setup.solid.fraction.len() != g.len() {
            return Err(CfdError::Geometry("solid field does not match grid".into()));
        }
        if !(setup.gas.gamma > 1.0) {
            return Err(CfdError::Parameter(format!("gamma = {}", setup.gas.gamma)));
        }
        let solid = pad_solid(&setup.solid, &g);
        let mut s = EulerSolver {
            state: State::new(&g),
            u1: vec![[0.0; 4]; g.glen()],
            w: vec![[0.0; 4]; g.glen()],
            rhs: vec![[0.0; 4]; g.glen()],
            mask: vec![false; g.glen()],
            solid,
            info: StepInfo { step: 0, time: 0.0, dt: 0.0, residual: f64::NAN,
                             converged: false, floor_activations: 0 },
            ledger: physics::FlipLedger::default(),
            residual_ref: f64::NAN,
            pending_geometry: None,
            setup,
        };
        s.reset_field();
        Ok(s)
    }

    /// Ambient everywhere, then the quasi-1D overlay if enabled.
    fn reset_field(&mut self) {
        let g = self.setup.grid;
        let a = self.setup.ambient;
        let rho_a = a.p / a.t; // p = rho*T, R = 1
        let ua = prim_to_cons([rho_a, 0.0, 0.0, a.p], self.setup.gas.gamma);
        for c in self.state.u_old.iter_mut() { *c = ua; }
        if self.setup.numerics.quasi1d_init {
            physics::quasi1d_init(&mut self.state.u_old, &g, &self.setup.solid,
                                  &self.setup.gas, &self.setup.chamber, &self.setup.ambient);
        }
        self.state.u_new.copy_from_slice(&self.state.u_old);
    }

    /// Overwrite the interior with canonical primitives `[rho, u_z, u_r, p]`.
    /// Init hook for the acceptance tests (Sod strips, free streams); not part
    /// of the `Solver` trait.
    pub fn set_initial(&mut self, f: impl Fn(usize, usize) -> Prim) {
        let g = self.setup.grid;
        for ir in 0..g.nr {
            for iz in 0..g.nz {
                let u = prim_to_cons(f(iz, ir), self.setup.gas.gamma);
                self.state.u_old[g.gidx(iz as isize, ir as isize)] = u;
            }
        }
        self.state.u_new.copy_from_slice(&self.state.u_old);
        self.info.time = 0.0;
        self.info.step = 0;
        self.residual_ref = f64::NAN;
    }

    pub fn setup(&self) -> &SolveSetup { &self.setup }
}

impl Solver for EulerSolver {
    fn step(&mut self) -> Result<StepInfo> {
        let g = self.setup.grid;
        let gas = self.setup.gas;
        let num = self.setup.numerics;

        // Geometry edits drain HERE, never inside an RK stage.
        if let Some(new_solid) = self.pending_geometry.take() {
            physics::apply_geometry_change(&mut self.state.u_old, &g, &self.setup.solid,
                                           &new_solid, &gas, &self.setup.ambient,
                                           &mut self.ledger);
            self.solid = pad_solid(&new_solid, &g);
            self.setup.solid = new_solid;
            self.residual_ref = f64::NAN; // reset the convergence monitor
            self.info.step = 0;
        }

        physics::fill_ghosts(&mut self.state.u_old, &g, &self.solid, &gas,
                             &self.setup.chamber, &self.setup.ambient, &num);
        kernel::compute_primitives(&self.state.u_old, &mut self.w, gas.gamma);
        physics::carbuncle_mask(&self.w, &g, &mut self.mask);

        let wmax = kernel::max_wave_speed(&self.w, &self.solid, &g, gas.gamma);
        let dt = num.cfl / wmax;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(CfdError::Diverged {
                step: self.info.step,
                detail: format!("max wave speed {wmax}, dt {dt}"),
            });
        }
        let mut floors = self.info.floor_activations;

        // Stage 1: u1 = u0 + dt*rhs.
        self.rhs.iter_mut().for_each(|c| *c = [0.0; 4]);
        kernel::sweep_z(&self.w, &self.solid, &self.mask, &g, &num, &gas,
                        &mut self.rhs, &mut floors);
        kernel::sweep_r(&self.w, &self.solid, &self.mask, &g, &num, &gas,
                        &mut self.rhs, &mut floors);
        kernel::accumulate(&mut self.u1, &self.state.u_old, &self.rhs, dt, &self.solid);
        kernel::enforce_positivity(&mut self.u1, &gas, &mut floors);

        // Stage 2: u_new = 0.5*(u0 + u1 + dt*rhs).
        physics::fill_ghosts(&mut self.u1, &g, &self.solid, &gas,
                             &self.setup.chamber, &self.setup.ambient, &num);
        kernel::compute_primitives(&self.u1, &mut self.w, gas.gamma);
        self.rhs.iter_mut().for_each(|c| *c = [0.0; 4]);
        kernel::sweep_z(&self.w, &self.solid, &self.mask, &g, &num, &gas,
                        &mut self.rhs, &mut floors);
        kernel::sweep_r(&self.w, &self.solid, &self.mask, &g, &num, &gas,
                        &mut self.rhs, &mut floors);
        kernel::accumulate2(&mut self.state.u_new, &self.state.u_old, &self.u1,
                            &self.rhs, dt, &self.solid);
        kernel::enforce_positivity(&mut self.state.u_new, &gas, &mut floors);

        physics::apply_sponge(&mut self.state.u_new, &g, dt, &self.setup.ambient,
                              &gas, num.sponge_cells);

        // The residual must not read stale ghost garbage: make the ghost bands
        // identical before the reduction.
        copy_ghost_band(&self.state.u_old, &mut self.state.u_new, &g);
        let raw = kernel::density_residual_f64(&self.state.u_old, &self.state.u_new,
                                               &self.solid);
        self.state.swap();

        self.info.step += 1;
        self.info.time += dt as f64;
        self.info.dt = dt;
        self.info.floor_activations = floors;
        // Normalized by the step-10 value; NaN before step 10.
        if self.info.step == 10 { self.residual_ref = raw.max(f64::MIN_POSITIVE); }
        self.info.residual = if self.info.step >= 10 { raw / self.residual_ref }
                             else { f64::NAN };
        self.info.converged = self.info.step >= 10 && self.info.residual < RESIDUAL_CONVERGED;
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
        let g = self.setup.grid;
        let gamma = self.setup.gas.gamma;
        let prims: Vec<Prim> = (0..g.nr)
            .flat_map(|ir| (0..g.nz).map(move |iz| (iz, ir)))
            .map(|(iz, ir)| cons_to_prim(self.state.u_old[g.gidx(iz as isize, ir as isize)], gamma))
            .collect();
        snapshot_from_prims(&self.setup, &prims, self.info)
    }

    fn report(&self) -> Report {
        let g = self.setup.grid;
        let gamma = self.setup.gas.gamma as f64;
        let refs = self.setup.refs;
        let solid = &self.setup.solid;
        let p_a = self.setup.ambient.p as f64;

        // Open radius per column (docs/physics-reference.md §5) and the lip.
        let mut lip: Option<usize> = None;
        let mut r_open = vec![0.0f64; g.nz];
        for iz in 0..g.nz {
            let mut acc = 0.0f64;
            let mut any_solid = false;
            for ir in 0..g.nr {
                let idx = g.idx(iz, ir);
                acc += (1.0 - solid.fraction[idx] as f64) * g.r_center(ir) as f64;
                any_solid |= solid.is_solid(idx);
            }
            r_open[iz] = (2.0 * g.dr as f64 * acc).sqrt();
            if any_solid { lip = Some(iz); }
        }
        let empty = Report {
            mass_flow_kg_s: f64::NAN, thrust_n: f64::NAN, thrust_coefficient: f64::NAN,
            c_star_m_s: f64::NAN, discharge_coefficient: f64::NAN, exit_mach: f64::NAN,
            exit_pressure_pa: f64::NAN, exit_pressure_ratio: f64::NAN,
            ideal_exit_mach: f64::NAN, cells_per_throat_radius: f64::NAN,
            converged: self.info.converged, confidence: Confidence::NotConverged,
        };
        let Some(lip) = lip else { return empty; };
        let r_t = r_open[..=lip].iter().cloned().fold(f64::INFINITY, f64::min);
        let r_e = r_open[lip];
        if !(r_t > 0.0) || !(r_e > 0.0) { return empty; }

        // Exit-plane control surface (docs/physics-reference.md §11), f64.
        let mut mdot = 0.0f64;
        let mut thrust = 0.0f64;
        let mut mach_a = 0.0f64;
        let mut p_area = 0.0f64;
        let mut area = 0.0f64;
        for ir in 0..g.nr {
            let idx = g.idx(lip, ir);
            if solid.is_solid(idx) { continue; }
            let w = cons_to_prim(self.state.u_old[g.gidx(lip as isize, ir as isize)],
                                 gamma as Real);
            let da = 2.0 * std::f64::consts::PI * g.r_center(ir) as f64 * g.dr as f64;
            let (rho, uz, p) = (w[0] as f64, w[1] as f64, w[3] as f64);
            mdot += rho * uz * da;
            thrust += (rho * uz * uz + p - p_a) * da;
            let m = (uz * uz + (w[2] as f64).powi(2)).sqrt()
                / sound_speed(w, gamma as Real) as f64;
            mach_a += m * da;
            p_area += p * da;
            area += da;
        }
        if area <= 0.0 || mdot.abs() < 1e-30 { return empty; }
        let a_t = std::f64::consts::PI * r_t * r_t;
        let exit_mach = mach_a / area;
        let exit_p = p_area / area;
        let force_scale = refs.p_pa * refs.l_m * refs.l_m;
        let mdot_si = mdot * refs.rho_kg_m3 * refs.u_m_s * refs.l_m * refs.l_m;
        let a_t_si = a_t * refs.l_m * refs.l_m;
        let c_star = refs.p_pa * a_t_si / mdot_si;
        let gm = gamma;
        let c_star_ideal = (gm * self.setup.gas.r_specific_si * refs.t_k).sqrt()
            / (gm * (2.0 / (gm + 1.0)).powf((gm + 1.0) / (2.0 * (gm - 1.0))));
        let pr = exit_p / p_a;
        let sep_threshold = (1.88 * exit_mach - 1.0).powf(-0.64).max(0.40);
        let n_throat = r_t / g.dr as f64;
        let confidence = if !self.info.converged { Confidence::NotConverged }
            else if pr < sep_threshold { Confidence::SeparationLikely }
            else if n_throat < 20.0 { Confidence::Underresolved }
            else { Confidence::Valid };
        Report {
            mass_flow_kg_s: mdot_si,
            thrust_n: thrust * force_scale,
            thrust_coefficient: thrust / a_t, // p0 = 1 non-dimensional
            c_star_m_s: c_star,
            discharge_coefficient: c_star_ideal / c_star,
            exit_mach,
            exit_pressure_pa: exit_p * refs.p_pa,
            exit_pressure_ratio: pr,
            ideal_exit_mach: physics::mach_from_area_ratio((r_e / r_t).powi(2), gamma, true),
            cells_per_throat_radius: n_throat,
            converged: self.info.converged,
            confidence,
        }
    }

    fn set_ambient(&mut self, a: Ambient) {
        // Never resets the field: the nozzle interior is supersonic, only the
        // plume re-equilibrates. See docs/physics-reference.md §9.
        self.setup.ambient = a;
        self.residual_ref = f64::NAN;
        self.info.step = 0;
    }

    fn set_geometry(&mut self, solid: Arc<SolidField>) -> Result<()> {
        if solid.grid != self.setup.grid || solid.fraction.len() != self.setup.grid.len() {
            return Err(CfdError::Geometry("solid field does not match grid".into()));
        }
        self.pending_geometry = Some(solid);
        Ok(())
    }

    fn set_numerics(&mut self, n: Numerics) { self.setup.numerics = n; }

    fn step_count(&self) -> u64 { self.info.step }
}

/// Threshold the solid fractions and replicate the edge into the ghost band.
fn pad_solid(solid: &SolidField, g: &cfd_contract::Grid) -> Vec<bool> {
    let mut out = vec![false; g.glen()];
    let ng = NG as isize;
    for ir in -ng..(g.nr as isize + ng) {
        for iz in -ng..(g.nz as isize + ng) {
            let cz = iz.clamp(0, g.nz as isize - 1) as usize;
            let cr = ir.clamp(0, g.nr as isize - 1) as usize;
            out[g.gidx(iz, ir)] = solid.is_solid(g.idx(cz, cr));
        }
    }
    out
}

/// Copy src's ghost band into dst, leaving dst's interior untouched.
fn copy_ghost_band(src: &[Cons], dst: &mut [Cons], g: &cfd_contract::Grid) {
    let ng = NG as isize;
    for ir in -ng..(g.nr as isize + ng) {
        for iz in -ng..(g.nz as isize + ng) {
            let interior = iz >= 0 && iz < g.nz as isize && ir >= 0 && ir < g.nr as isize;
            if !interior {
                let i = g.gidx(iz, ir);
                dst[i] = src[i];
            }
        }
    }
}

/// The ONLY place non-dimensional -> SI happens. Shared by `EulerSolver` and
/// `MockSolver`. `prims` is interior-only, `grid.len()` long, canonical
/// `[rho, u_z, u_r, p]`.
pub(crate) fn snapshot_from_prims(setup: &SolveSetup, prims: &[Prim], info: StepInfo) -> Snapshot {
    let g = setup.grid;
    let refs = setup.refs;
    let gamma = setup.gas.gamma;
    let n = g.len();
    let mut fields = vec![vec![0.0f32; n]; FieldKind::ALL.len()];
    for i in 0..n {
        if setup.solid.is_solid(i) { continue; }
        let [rho, uz, ur, p] = prims[i];
        let speed_nd = (uz * uz + ur * ur).sqrt();
        let a = (gamma * p / rho).sqrt();
        fields[FieldKind::Density as usize][i] = rho * refs.rho_kg_m3 as f32;
        fields[FieldKind::Pressure as usize][i] = p * refs.p_pa as f32;
        fields[FieldKind::Temperature as usize][i] = (p / rho) * refs.t_k as f32;
        fields[FieldKind::Mach as usize][i] = speed_nd / a;
        fields[FieldKind::VelocityZ as usize][i] = uz * refs.u_m_s as f32;
        fields[FieldKind::VelocityR as usize][i] = ur * refs.u_m_s as f32;
        fields[FieldKind::Speed as usize][i] = speed_nd * refs.u_m_s as f32;
    }
    // Schlieren: |grad rho| by central differences, normalized to [0, 1].
    let mut smax = 0.0f32;
    for ir in 0..g.nr {
        for iz in 0..g.nz {
            let i = g.idx(iz, ir);
            if setup.solid.is_solid(i) { continue; }
            let rho_at = |jz: usize, jr: usize| -> f32 {
                let j = g.idx(jz, jr);
                if setup.solid.is_solid(j) { prims[i][0] } else { prims[j][0] }
            };
            let dz = (rho_at((iz + 1).min(g.nz - 1), ir) - rho_at(iz.saturating_sub(1), ir))
                / (2.0 * g.dz);
            let dr = (rho_at(iz, (ir + 1).min(g.nr - 1)) - rho_at(iz, ir.saturating_sub(1)))
                / (2.0 * g.dr);
            let s = (dz * dz + dr * dr).sqrt();
            fields[FieldKind::Schlieren as usize][i] = s;
            smax = smax.max(s);
        }
    }
    if smax > 0.0 {
        for s in fields[FieldKind::Schlieren as usize].iter_mut() { *s /= smax; }
    }
    // Ranges over FLUID cells only.
    let ranges = fields.iter().map(|f| {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for i in 0..n {
            if setup.solid.is_solid(i) { continue; }
            lo = lo.min(f[i]);
            hi = hi.max(f[i]);
        }
        if lo > hi { (0.0, 0.0) } else { (lo, hi) }
    }).collect();
    Snapshot {
        grid: g,
        step: info.step,
        time_s: info.time * refs.time_s,
        residual: info.residual,
        converged: info.converged,
        solid: setup.solid.clone(),
        fields,
        ranges,
    }
}
