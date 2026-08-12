//! The contract. **Frozen.** No session may edit this crate. If you believe it is wrong,
//! print `CONTRACT CHANGE REQUEST:` with the exact diff and stop.
//!
//! # Conventions
//!
//! Violating one of these is the most likely source of a silent bug at merge.
//!
//! - **Non-dimensional everywhere inside the solver.** Chamber-referenced: `L_ref = r_t`,
//!   `p_ref = p0`, `rho_ref = rho0`, `u_ref = sqrt(p0/rho0)`, `T_ref = T0`. So `R = 1`, `p = rho*T`,
//!   and the chamber state is exactly `(1, 0, 0, 1)`. SI appears only in `Snapshot`, `Report`
//!   and the UI. `RefScales` does the conversion and nothing else may.
//! - **`f32` in the hot loop** (`pub type Real = f32`). **Every reduction accumulates in `f64`**
//!   — mass, momentum, energy, thrust integrals, L1 norms, residuals. An f32 sum over 64k cells
//!   contributes more noise than the Sod pass threshold.
//! - **Row-major, z contiguous.** Interior index `idx = ir*nz + iz`. Padded index
//!   `gidx = (ir+NG)*(nz+2*NG) + (iz+NG)`. Use `Grid::idx` and `Grid::gidx`; never open-code either.
//! - **Cell centres at `r = (ir + 0.5)*dr`.** Never zero. The lower face of row 0 is at
//!   `r = 0`, which is why it carries zero flux and why the axisymmetric source is finite.
//! - **Ghost cells are private to `cfd-core`.** Every array crossing a crate boundary is
//!   exactly `grid.len()` long, interior only.
//! - **Ping-pong.** Read `u_old` immutably, write `u_new` through `par_chunks_mut` over
//!   rows. Never mutate in place.
//! - **One error type**, `CfdError`, defined here. No crate defines its own and none
//!   returns `anyhow::Result` in its public API.
//! - **`Prim` is rotated to the face.** `Prim = [rho, u_n, u_t, p]` where `n` is the sweep
//!   direction: z-sweep `u_n = u_z`, `u_t = u_r`; r-sweep `u_n = u_r`, `u_t = u_z`. `hllc_flux`
//!   returns a `Cons` in that same rotated frame; the caller rotates back. `compute_primitives`
//!   produces the **canonical unrotated** `[rho, u_z, u_r, p]`; the sweeps rotate at use.

pub mod kernels;

pub type Real = f32;

pub type Cons = [Real; 4]; // [rho, rho*u_z, rho*u_r, E]
pub type Prim = [Real; 4]; // [rho, u_n, u_t, p] — rotated to the sweep direction. See conventions.

pub const NG: usize = 2; // ghost width. MUSCL needs 2. Baked into the padded layout.
pub const SOLID_THRESHOLD: f32 = 0.5; // cells at/above this fraction are frozen, never updated
pub const CFL_DEFAULT: Real = 0.4;
// Non-dimensional floors. See docs/physics-reference.md §5.
pub const RHO_MIN: Real = 1e-8;
pub const P_MIN_ABS: Real = 1e-8;

#[derive(Debug, thiserror::Error)]
pub enum CfdError {
    #[error("invalid geometry: {0}")] Geometry(String),
    #[error("invalid grid: {0}")] Grid(String),
    #[error("invalid parameter: {0}")] Parameter(String),
    #[error("solver diverged at step {step}: {detail}")] Diverged { step: u64, detail: String },
    #[error("io: {0}")] Io(#[from] std::io::Error),
}
pub type Result<T> = std::result::Result<T, CfdError>;

/// Uniform, anisotropic (dz != dr), axisymmetric (z, r) grid of cell centres.
/// All lengths non-dimensional, in units of throat radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid { pub nz: usize, pub nr: usize, pub dz: Real, pub dr: Real }

impl Grid {
    /// Interior cell count. Every slice crossing a crate boundary is this long.
    pub fn len(&self) -> usize { self.nz * self.nr }
    pub fn idx(&self, iz: usize, ir: usize) -> usize { ir * self.nz + iz } // row-major, z contiguous
    pub fn snz(&self) -> usize { self.nz + 2 * NG } // padded row stride, ghosts both sides
    pub fn snr(&self) -> usize { self.nr + 2 * NG }
    pub fn glen(&self) -> usize { self.snz() * self.snr() }
    /// Padded index. Accepts negative interior coordinates for ghost access.
    pub fn gidx(&self, iz: isize, ir: isize) -> usize {
        ((ir + NG as isize) as usize) * self.snz() + ((iz + NG as isize) as usize)
    }
    pub fn r_center(&self, ir: usize) -> Real { (ir as Real + 0.5) * self.dr } // row 0: dr/2, never 0
    pub fn r_face(&self, ir: usize) -> Real { ir as Real * self.dr } // row 0: exactly 0
    /// Cell volume / (2*pi): r_c*dr*dz. Exact — r_c is the mean of the face radii, so this
    /// is the true cylindrical shell volume. ONE implementation; never write your own.
    pub fn cell_vol(&self, ir: usize) -> f64 {
        self.r_center(ir) as f64 * self.dr as f64 * self.dz as f64
    }
}

// ---- the five abort-plan flags. Every one honoured by the solver from day one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum Geometry { Axisymmetric, Planar }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum Reconstruction { FirstOrder, Muscl }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum WallMode { Mirror, ColumnReflect }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum SolverKind { Mock, Real }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum FluxMode { Hllc, HllRadial, Hll }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum Limiter { None, Minmod, VanLeer }

#[derive(Debug, Clone, Copy)]
pub struct GasModel { pub gamma: Real, pub r_specific_si: f64 }

/// Non-dimensional <-> SI. The only place a conversion may live.
#[derive(Debug, Clone, Copy)]
pub struct RefScales {
    pub l_m: f64, pub p_pa: f64, pub rho_kg_m3: f64, pub u_m_s: f64, pub t_k: f64, pub time_s: f64,
}
impl RefScales {
    pub fn from_chamber(r_t_m: f64, p0_pa: f64, t0_k: f64, gas: &GasModel) -> Self {
        let rho0 = p0_pa / (gas.r_specific_si * t0_k);
        let u = (p0_pa / rho0).sqrt();
        RefScales { l_m: r_t_m, p_pa: p0_pa, rho_kg_m3: rho0, u_m_s: u, t_k: t0_k, time_s: r_t_m / u } } }

/// The only thing the solver knows about geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct SolidField { pub grid: Grid, pub fraction: Vec<f32> }
impl SolidField {
    pub fn is_solid(&self, idx: usize) -> bool { self.fraction[idx] >= SOLID_THRESHOLD }
    pub fn empty(grid: Grid) -> Self { SolidField { grid, fraction: vec![0.0; grid.len()] } }
}

// Non-dimensional. Ambient is what the altitude slider moves; Chamber is (1, 1) by construction.
#[derive(Debug, Clone, Copy)] pub struct Ambient { pub p: Real, pub t: Real }
#[derive(Debug, Clone, Copy)] pub struct Chamber { pub p0: Real, pub t0: Real }

#[derive(Debug, Clone, Copy)]
pub struct Numerics {
    pub cfl: Real,
    pub limiter: Limiter,
    pub geometry: Geometry,
    pub reconstruction: Reconstruction,
    pub wall_mode: WallMode,
    pub flux_mode: FluxMode,
    pub quasi1d_init: bool,
    pub sponge_cells: usize, // 24
}
impl Default for Numerics {
    fn default() -> Self {
        Numerics { cfl: CFL_DEFAULT, limiter: Limiter::Minmod, geometry: Geometry::Axisymmetric,
                   reconstruction: Reconstruction::Muscl, wall_mode: WallMode::Mirror,
                   flux_mode: FluxMode::Hllc, quasi1d_init: true, sponge_cells: 24 } } }

#[derive(Debug, Clone)]
pub struct SolveSetup {
    pub grid: Grid,
    pub solid: std::sync::Arc<SolidField>,
    pub gas: GasModel,
    pub chamber: Chamber,
    pub ambient: Ambient,
    pub numerics: Numerics,
    pub refs: RefScales,
}

/// Ping-pong state. Both buffers are glen() long, including ghosts.
pub struct State { pub u_old: Vec<Cons>, pub u_new: Vec<Cons> }
impl State {
    pub fn new(g: &Grid) -> Self {
        State { u_old: vec![[0.0; 4]; g.glen()], u_new: vec![[0.0; 4]; g.glen()] }
    }
    pub fn swap(&mut self) { std::mem::swap(&mut self.u_old, &mut self.u_new) }
}

#[derive(Debug, Clone, Copy)]
pub struct StepInfo {
    pub step: u64,
    pub time: f64, // non-dimensional
    pub dt: Real,
    pub residual: f64, // L2 of the density update / its step-10 value. NaN before step 10.
    pub converged: bool,
    pub floor_activations: u64, // cumulative. Nonzero: every downstream number is un-auditable.
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum FieldKind {
    Density = 0, Pressure = 1, Temperature = 2, Mach = 3,
    VelocityZ = 4, VelocityR = 5, Speed = 6, Schlieren = 7,
}
impl FieldKind {
    /// Discriminants ARE the indices into Snapshot::fields. Do not reorder.
    pub const ALL: [FieldKind; 8] = [
        FieldKind::Density, FieldKind::Pressure, FieldKind::Temperature, FieldKind::Mach,
        FieldKind::VelocityZ, FieldKind::VelocityR, FieldKind::Speed, FieldKind::Schlieren,
    ];
    pub fn label(self) -> &'static str {
        match self {
            FieldKind::Density => "Density [kg/m³]", FieldKind::Pressure => "Pressure [Pa]",
            FieldKind::Temperature => "Temperature [K]", FieldKind::Mach => "Mach",
            FieldKind::VelocityZ => "Velocity z [m/s]", FieldKind::VelocityR => "Velocity r [m/s]",
            FieldKind::Speed => "Speed [m/s]", FieldKind::Schlieren => "Schlieren",
        }
    }
    pub fn is_signed(self) -> bool { matches!(self, FieldKind::VelocityZ | FieldKind::VelocityR) } }

/// Immutable copy for display. SI. f32. Interior cells only, grid.len() long.
/// The ONLY place non-dimensional -> SI happens.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub grid: Grid,
    pub step: u64,
    pub time_s: f64,
    pub residual: f64,
    pub converged: bool,
    pub solid: std::sync::Arc<SolidField>,
    pub fields: Vec<Vec<f32>>, // indexed by FieldKind as usize. Exactly 8, each grid.len() long.
    pub ranges: Vec<(f32, f32)>, // per-field (min, max) over FLUID cells only
}
impl Snapshot {
    pub fn field(&self, k: FieldKind) -> &[f32] { &self.fields[k as usize] }
    pub fn range(&self, k: FieldKind) -> (f32, f32) { self.ranges[k as usize] }
    pub fn sample(&self, k: FieldKind, iz: usize, ir: usize) -> f32 {
        self.fields[k as usize][self.grid.idx(iz, ir)]
    }
    /// Axis row (ir = 0), one value per column.
    pub fn centerline(&self, k: FieldKind) -> Vec<f32> {
        (0..self.grid.nz).map(|iz| self.sample(k, iz, 0)).collect()
    }
}

/// Integrated engineering output, SI. One definition each — see
/// docs/physics-reference.md §11. Validation asserts those definitions only.
#[derive(Debug, Clone, Copy)]
pub struct Report {
    pub mass_flow_kg_s: f64,
    pub thrust_n: f64,
    pub thrust_coefficient: f64,
    pub c_star_m_s: f64,
    pub discharge_coefficient: f64,
    pub exit_mach: f64,
    pub exit_pressure_pa: f64,
    pub exit_pressure_ratio: f64,
    /// Quasi-1D isentropic value at this area ratio. A correct 2D bell SHOULD miss this
    /// by 1-2%; that gap is the feature, not an error.
    pub ideal_exit_mach: f64,
    pub cells_per_throat_radius: f64,
    pub converged: bool,
    pub confidence: Confidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence { Valid, SeparationLikely, Underresolved, NotConverged }

/// Object safe on purpose: the app holds a Box<dyn Solver>.
pub trait Solver: Send {
    fn step(&mut self) -> Result<StepInfo>;
    fn run(&mut self, max_steps: u64) -> Result<StepInfo>;
    fn snapshot(&self) -> Snapshot;
    fn report(&self) -> Report;
    /// Changes ambient WITHOUT discarding the field. Load-bearing: the nozzle interior is
    /// supersonic, so only the plume re-equilibrates; a reset here would make the altitude
    /// slider unusable.
    fn set_ambient(&mut self, a: Ambient);
    /// New geometry mid-run. Must run the flip ledger and BFS refill — physics-reference §3.
    fn set_geometry(&mut self, solid: std::sync::Arc<SolidField>) -> Result<()>;
    fn set_numerics(&mut self, n: Numerics);
    fn step_count(&self) -> u64;
}

pub enum SolverCommand {
    Pause(bool), SingleStep, Reset, Turbo(u32),
    SetAmbient(Ambient), SetGeometry(std::sync::Arc<SolidField>), SetNumerics(Numerics),
}
