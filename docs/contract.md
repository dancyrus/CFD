# The Contract

The single artifact that lets four Claude Code sessions build four crates at once without talking to each other. The coordinator session transcribes this into real files during the blocking phase. After that it is **frozen**: no session may edit `cfd-contract/`, `cfd-core/src/lib.rs`, or `cfd-core/src/step.rs`.

Signatures below are the agreement. Sessions code against these names, and they are correct today — no session has to wait to find out what a type is called.

---

## Conventions

Violating one of these is the most likely source of a silent bug at merge.

- **Non-dimensional everywhere inside the solver.** Chamber-referenced: `L_ref = r_t`, `p_ref = p0`, `rho_ref = rho0`, `u_ref = sqrt(p0/rho0)`, `T_ref = T0`. So `R = 1`, `p = rho*T`, and the chamber state is exactly `(1, 0, 0, 1)`. SI appears only in `Snapshot`, `Report` and the UI. `RefScales` does the conversion and nothing else may.
- **`f32` in the hot loop** (`pub type Real = f32`). **Every reduction accumulates in `f64`** — mass, momentum, energy, thrust integrals, L1 norms, residuals. An f32 sum over 64k cells contributes more noise than the Sod pass threshold.
- **Row-major, z contiguous.** Interior index `idx = ir*nz + iz`. Padded index `gidx = (ir+NG)*(nz+2*NG) + (iz+NG)`. Use `Grid::idx` and `Grid::gidx`; never open-code either.
- **Cell centres at `r = (ir + 0.5)*dr`.** Never zero. The lower face of row 0 is at `r = 0`, which is why it carries zero flux and why the axisymmetric source is finite.
- **Ghost cells are private to `cfd-core`.** Every array crossing a crate boundary is exactly `grid.len()` long, interior only.
- **Ping-pong.** Read `u_old` immutably, write `u_new` through `par_chunks_mut` over rows. Never mutate in place. This is the difference between working rayon code and forty minutes of borrow-checker errors.
- **One error type**, `CfdError`, defined here. No crate defines its own and none returns `anyhow::Result` in its public API.
- **`Prim` is rotated to the face.** `Prim = [rho, u_n, u_t, p]` where `n` is the sweep direction. For a z-sweep `u_n = u_z`, `u_t = u_r`. For an r-sweep `u_n = u_r`, `u_t = u_z`. `hllc_flux` returns a `Cons` in that same rotated frame; the caller rotates back. `compute_primitives` produces the **canonical unrotated** form `[rho, u_z, u_r, p]`; the sweeps rotate at the point of use.

---

## `cfd-contract/src/lib.rs`

Target: under 250 lines. Later sessions will have their context compacted and must be able to re-read this cheaply.

```rust
pub type Real = f32;

/// [rho, rho*u_z, rho*u_r, E]
pub type Cons = [Real; 4];
/// [rho, u_n, u_t, p] — rotated to the sweep direction. See conventions.
pub type Prim = [Real; 4];

/// Ghost width. MUSCL needs 2. Baked into the padded layout.
pub const NG: usize = 2;
/// Cells at or above this solid fraction are frozen and never updated.
pub const SOLID_THRESHOLD: f32 = 0.5;
pub const CFL_DEFAULT: Real = 0.4;
/// Non-dimensional floors. See docs/physics-reference.md §5.
pub const RHO_MIN: Real = 1e-8;
pub const P_MIN_ABS: Real = 1e-6;

#[derive(Debug, thiserror::Error)]
pub enum CfdError {
    #[error("invalid geometry: {0}")]      Geometry(String),
    #[error("invalid grid: {0}")]          Grid(String),
    #[error("invalid parameter: {0}")]     Parameter(String),
    #[error("solver diverged at step {step}: {detail}")]
                                           Diverged { step: u64, detail: String },
    #[error("io: {0}")]                    Io(#[from] std::io::Error),
}
pub type Result<T> = std::result::Result<T, CfdError>;

/// Uniform, anisotropic (dz != dr), axisymmetric (z, r) grid of cell centres.
/// All lengths non-dimensional, in units of throat radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid { pub nz: usize, pub nr: usize, pub dz: Real, pub dr: Real }

impl Grid {
    /// Interior cell count. Every slice crossing a crate boundary is this long.
    pub fn len(&self) -> usize { self.nz * self.nr }
    /// Interior index. Row-major, z contiguous.
    pub fn idx(&self, iz: usize, ir: usize) -> usize { ir * self.nz + iz }
    /// Padded row stride, including ghosts on both sides.
    pub fn snz(&self) -> usize { self.nz + 2 * NG }
    pub fn snr(&self) -> usize { self.nr + 2 * NG }
    pub fn glen(&self) -> usize { self.snz() * self.snr() }
    /// Padded index. Accepts negative interior coordinates for ghost access.
    pub fn gidx(&self, iz: isize, ir: isize) -> usize {
        ((ir + NG as isize) as usize) * self.snz() + ((iz + NG as isize) as usize)
    }
    /// Cell-centre radius. Row 0 returns dr/2, never 0.
    pub fn r_center(&self, ir: usize) -> Real { (ir as Real + 0.5) * self.dr }
    /// Lower face radius of row ir. Row 0 returns exactly 0.
    pub fn r_face(&self, ir: usize) -> Real { ir as Real * self.dr }
    /// Cell volume / (2*pi): r_c * dr * dz. Exact — r_c is the arithmetic mean
    /// of the two face radii, so this is the true cylindrical shell volume.
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
    pub l_m: f64, pub p_pa: f64, pub rho_kg_m3: f64,
    pub u_m_s: f64, pub t_k: f64, pub time_s: f64,
}
impl RefScales {
    pub fn from_chamber(r_t_m: f64, p0_pa: f64, t0_k: f64, gas: &GasModel) -> Self { unimplemented!() }
}

/// The only thing the solver knows about geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct SolidField { pub grid: Grid, pub fraction: Vec<f32> }
impl SolidField {
    pub fn is_solid(&self, idx: usize) -> bool { self.fraction[idx] >= SOLID_THRESHOLD }
    pub fn empty(grid: Grid) -> Self { unimplemented!() }
}

/// Non-dimensional ambient state. This is what the altitude slider moves.
#[derive(Debug, Clone, Copy)] pub struct Ambient { pub p: Real, pub t: Real }
/// Non-dimensional chamber stagnation state. (1.0, 1.0) by construction.
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
    pub sponge_cells: usize,      // 24
}
impl Default for Numerics { /* cfl 0.4, Minmod, Axisymmetric, Muscl, Mirror, Hllc, true, 24 */ }

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
    pub fn new(grid: &Grid) -> Self { unimplemented!() }
    pub fn swap(&mut self) { std::mem::swap(&mut self.u_old, &mut self.u_new) }
}

#[derive(Debug, Clone, Copy)]
pub struct StepInfo {
    pub step: u64,
    pub time: f64,              // non-dimensional
    pub dt: Real,
    /// L2 of the density update, normalized by its step-10 value. NaN before step 10.
    pub residual: f64,
    pub converged: bool,
    /// Cumulative. Nonzero means every downstream number is un-auditable.
    pub floor_activations: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum FieldKind {
    Density = 0, Pressure = 1, Temperature = 2, Mach = 3,
    VelocityZ = 4, VelocityR = 5, Speed = 6, Schlieren = 7,
}
impl FieldKind {
    /// Discriminants ARE the indices into Snapshot::fields. Do not reorder.
    pub const ALL: [FieldKind; 8] = [ /* in discriminant order */ ];
    pub fn label(self) -> &'static str;   // e.g. "Pressure [Pa]"
    pub fn is_signed(self) -> bool;
}

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
    /// Indexed by FieldKind as usize. Exactly 8, each grid.len() long.
    pub fields: Vec<Vec<f32>>,
    /// Per-field (min, max) over FLUID cells only.
    pub ranges: Vec<(f32, f32)>,
}
impl Snapshot {
    pub fn field(&self, k: FieldKind) -> &[f32];
    pub fn range(&self, k: FieldKind) -> (f32, f32);
    pub fn sample(&self, k: FieldKind, iz: usize, ir: usize) -> f32;
    pub fn centerline(&self, k: FieldKind) -> Vec<f32>;
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
    /// Quasi-1D isentropic value at this area ratio. A correct 2D bell SHOULD
    /// miss this by 1-2%; that gap is the feature, not an error.
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
    /// Changes ambient WITHOUT discarding the field. Load-bearing: the nozzle
    /// interior is supersonic, so only the plume re-equilibrates. Resetting here
    /// would make the altitude slider unusable.
    fn set_ambient(&mut self, a: Ambient);
    /// New geometry mid-run. Implementations must run the flip ledger and BFS
    /// refill — see docs/physics-reference.md §3.
    fn set_geometry(&mut self, solid: std::sync::Arc<SolidField>) -> Result<()>;
    fn set_numerics(&mut self, n: Numerics);
    fn step_count(&self) -> u64;
}

pub enum SolverCommand {
    Pause(bool), SingleStep, Reset,
    SetAmbient(Ambient),
    SetGeometry(std::sync::Arc<SolidField>),
    SetNumerics(Numerics),
    Turbo(u32),
}
```

---

## `cfd-contract/src/kernels.rs`

Pure, stateless, unit-testable in isolation. Written by the coordinator so both solver sessions share a compiled artifact rather than an agreement about behaviour.

```rust
pub fn minmod(a: Real, b: Real) -> Real;

pub fn cons_to_prim(u: Cons, gamma: Real) -> Prim;   // canonical [rho, u_z, u_r, p]
pub fn prim_to_cons(w: Prim, gamma: Real) -> Cons;
pub fn sound_speed(w: Prim, gamma: Real) -> Real;

/// Toro's S_M formulation. Direction-agnostic: caller rotates so the face normal
/// is +n. Returns Cons in the SAME rotated frame.
pub fn hllc_flux(ql: Prim, qr: Prim, gamma: Real) -> Cons;

/// Same contract. Used where the carbuncle sensor fires.
pub fn hll_flux(ql: Prim, qr: Prim, gamma: Real) -> Cons;

/// s = four consecutive cells straddling face i+1/2: [i-1, i, i+1, i+2].
/// Any true in `solid` drops that side to first order. Returns (left, right)
/// face states. Reconstruction::FirstOrder returns the cell averages unchanged.
pub fn muscl_face_states(
    s: [Prim; 4], solid: [bool; 4], recon: Reconstruction, lim: Limiter,
) -> (Prim, Prim);
```

Coordinator's tests for these, which must pass before the fan-out:

- `hllc_flux` across the Sod interface reproduces the exact star state `p* = 0.3031301781`, `u* = 0.9274526200`.
- `muscl_face_states` on a linear field reproduces that field exactly.
- `prim_to_cons(cons_to_prim(u))` round-trips to 1e-6 relative.

---

## `cfd-core/src/step.rs` — frozen, coordinator writes it

SSP-RK2. Calls functions that do not exist yet; the coordinator declares them with `todo!()` bodies in `kernel.rs` and `physics.rs`.

```
step():
  physics::fill_ghosts(u_old, ...)
  kernel::compute_primitives(u_old, w, gamma)
  physics::carbuncle_mask(w, grid, &mut mask)
  dt = numerics.cfl / kernel::max_wave_speed(w, solid, grid, gamma)

  // stage 1
  kernel::sweep_z(w, solid, mask, grid, num, gas, &mut rhs, &mut floors)
  kernel::sweep_r(w, solid, mask, grid, num, gas, &mut rhs, &mut floors)   // adds axisym source
  kernel::accumulate(&mut u_1, u_old, &rhs, dt, solid)                     // u1 = u0 + dt*rhs
  kernel::enforce_positivity(&mut u_1, gas, &mut floors)

  // stage 2
  physics::fill_ghosts(u_1, ...)
  kernel::compute_primitives(u_1, w, gamma)
  kernel::sweep_z(...); kernel::sweep_r(...)
  kernel::accumulate2(&mut u_new, u_old, u_1, &rhs, dt, solid)             // 0.5*(u0 + u1 + dt*rhs)
  kernel::enforce_positivity(&mut u_new, gas, &mut floors)

  physics::apply_sponge(&mut u_new, grid, dt, ambient, gas, num.sponge_cells)
  residual = kernel::density_residual_f64(u_old, u_new, solid)
  state.swap()
```

---

## `cfd-core/src/kernel.rs` — session A owns it

```rust
pub fn compute_primitives(u: &[Cons], w: &mut [Prim], gamma: Real);
pub fn max_wave_speed(w: &[Prim], solid: &[bool], g: &Grid, gamma: Real) -> Real;

pub fn sweep_z(w: &[Prim], solid: &[bool], mask: &[bool], g: &Grid,
               n: &Numerics, gas: &GasModel, rhs: &mut [Cons], floors: &mut u64);

/// Includes the r-weighted radial fluxes AND the axisymmetric pressure source,
/// written inside a single bracket. See docs/physics-reference.md §1.
pub fn sweep_r(w: &[Prim], solid: &[bool], mask: &[bool], g: &Grid,
               n: &Numerics, gas: &GasModel, rhs: &mut [Cons], floors: &mut u64);

pub fn accumulate(out: &mut [Cons], u0: &[Cons], rhs: &[Cons], dt: Real, solid: &[bool]);
pub fn accumulate2(out: &mut [Cons], u0: &[Cons], u1: &[Cons], rhs: &[Cons], dt: Real, solid: &[bool]);
pub fn enforce_positivity(u: &mut [Cons], gas: &GasModel, floors: &mut u64);
pub fn density_residual_f64(u0: &[Cons], u1: &[Cons], solid: &[bool]) -> f64;
```

At a fluid/solid face the sweeps call `crate::physics::wall_flux_z` / `wall_flux_r`. Session A codes against those signatures and does not implement them.

---

## `cfd-core/src/physics.rs` — session B owns it

```rust
/// Wall flux at a fluid/solid face. `w` is the fluid cell's CANONICAL primitives.
/// `sgn` is +1 when the fluid is on the low side of the face, -1 on the high side.
/// Returns [0, sgn*ps, 0, 0] — mass and energy flux are bit-exactly zero.
/// Special-cased, NOT computed by calling hllc_flux with a mirrored state.
pub fn wall_flux_z(w: Prim, sgn: Real, gamma: Real) -> Cons;
/// Returns [0, 0, sgn*ps, 0].
pub fn wall_flux_r(w: Prim, sgn: Real, gamma: Real) -> Cons;

/// Axis mirror, stagnation inlet, supersonic/subsonic outflow, radial far field.
pub fn fill_ghosts(u: &mut [Cons], g: &Grid, solid: &[bool], gas: &GasModel,
                   chamber: &Chamber, ambient: &Ambient, n: &Numerics);

/// dt-based, sigma_max = 12*a_ambient/L_sponge. NOT the per-step form.
pub fn apply_sponge(u: &mut [Cons], g: &Grid, dt: Real, ambient: &Ambient,
                    gas: &GasModel, cells: usize);

/// Omega = min/max of p over a +/-2 cell axial window; true where Omega < 0.7.
/// The caller uses HLL for the radial fluxes of cells i-1, i, i+1 where set.
pub fn carbuncle_mask(w: &[Prim], g: &Grid, mask: &mut [bool]);

/// Quasi-1D isentropic in the nozzle, ambient elsewhere, blended over 4 cells.
pub fn quasi1d_init(u: &mut [Cons], g: &Grid, solid: &SolidField,
                    gas: &GasModel, chamber: &Chamber, ambient: &Ambient);

/// Area-Mach inversion, NASA b4wind. Guard |ar - 1| < 1e-6 -> return 1.0.
pub fn mach_from_area_ratio(ar: f64, gamma: f64, supersonic: bool) -> f64;

#[derive(Default, Debug, Clone, Copy)]
pub struct FlipLedger { pub mass: f64, pub energy: f64 }

/// Flip bookkeeping plus BFS refill of newly opened cells at rest.
pub fn apply_geometry_change(u: &mut [Cons], g: &Grid, old: &SolidField,
                             new: &SolidField, gas: &GasModel,
                             ambient: &Ambient, ledger: &mut FlipLedger);
```

---

## `cfd-geom` — session C owns it

```rust
pub enum ContourKind {
    Conical { half_angle_deg: f64 },
    ParabolicBell { bell_percent: f64 },   // theta_n, theta_e from the Rao table
}

pub struct NozzleSpec {
    pub throat_radius_m: f64,
    pub area_ratio: f64,
    pub contraction_ratio: f64,        // 4.0
    pub converge_half_angle_deg: f64,  // 30.0
    pub throat_arc_up: f64,            // 1.5  (x r_t)
    pub throat_arc_down: f64,          // 0.382
    pub contour: ContourKind,
}
impl NozzleSpec {
    pub fn exit_radius_m(&self) -> f64;
    pub fn throat_area_m2(&self) -> f64;
    pub fn validate(&self) -> Result<()>;
}

/// Polyline in the (z, r) half-plane, increasing z, all r > 0. Also the EDITABLE
/// representation: dragging a control point edits `points` directly, so a
/// hand-edited wall and a generated wall are the same type and the solver
/// cannot tell them apart.
pub struct WallProfile { pub points: Vec<[f64; 2]>, pub throat_index: usize }
impl WallProfile {
    pub fn radius_at(&self, z: f64) -> Option<f64>;
    pub fn validate(&self) -> Result<()>;
}

pub fn generate_contour(spec: &NozzleSpec, samples: usize) -> Result<WallProfile>;
pub fn rao_angles(area_ratio: f64, bell_percent: f64) -> (f64, f64);

/// EXACT sub-cell area fractions. Point sampling is unacceptable.
pub fn rasterize(p: &WallProfile, g: &Grid, refs: &RefScales) -> Result<SolidField>;

/// Editor data model. No egui dependency.
pub struct Editor { /* control points, selection, hit radius */ }
impl Editor {
    pub fn from_profile(p: &WallProfile) -> Self;
    pub fn to_profile(&self) -> Result<WallProfile>;
    pub fn hit_test(&self, world: [f64; 2], tol: f64) -> Option<usize>;
    pub fn drag(&mut self, i: usize, world: [f64; 2]);
    pub fn insert(&mut self, world: [f64; 2]);
    pub fn remove(&mut self, i: usize);
}
```

---

## Predicted merge failures and what prevents each

| What breaks | Why it happens with agents | What stops it |
|---|---|---|
| Transposed image, plume points up | viz assumes (row, col) = (z, r); core writes (r, z) | `Grid::idx` is a concrete method, not a doc note. Only `cfd-ui` flips row order for display |
| Factor of 1000 | a session writes throat radius in mm because datasheets do | Unit suffix on every SI field name; everything else is non-dimensional by construction |
| Factor of 2π or r | two sessions independently write a cell volume; one folds in 2πr | `Grid::cell_vol` has a body here. One implementation, already written |
| Thrust disagreement | wall-pressure integral vs exit-plane control volume both defensible | `Report::thrust_n` names the exit-plane definition; validation asserts that only |
| Length mismatch panic | geometry includes ghosts, solver does not | "Ghost cells are private to cfd-core, every crossing array is `grid.len()`" plus `SolidField` carrying its own `Grid` |
| Three error types | every session reaches for `thiserror` | One `CfdError`, explicit no-local-errors rule |
| Enum reorder | someone alphabetizes `FieldKind` | Explicit discriminants plus `#[repr(usize)]` and a do-not-reorder note |
| Axis NaN on step 1 | `p/r` divides by a cell centre placed at 0 | `Grid::r_center` has a body here and returns `dr/2` at row 0 |
| Cargo.lock conflicts | four worktrees, four resolutions | Lockfile committed in the scaffold; no session runs `cargo update` |
| Faint axis artifact | flux difference and source accumulated separately in f32 | The single-bracket form is mandated in `step.rs`, which is frozen |
