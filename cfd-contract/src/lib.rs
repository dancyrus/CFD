//! The contract. Changes follow the CLAUDE.md contract-change rule: mirror
//! docs/contract.md and docs/physics-reference.md in the same commit, and
//! re-run the full acceptance ladder.
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
//! - **The grid is a tensor product of arbitrary cell-edge lists** (graded or uniform — the
//!   solver cannot tell and must not care why they are spaced that way). The lower face of
//!   row 0 is at `r = 0` always, which is why it carries zero flux and why the axisymmetric
//!   source is finite. `r_center` is the arithmetic mean of the two face radii (the volume
//!   radius); `r_centroid_g` is the shell volume centroid — reconstruction wants the centroid,
//!   the p/r balance never picks a radius at all (see docs/physics-reference.md §1).
//! - **Ghost cells are private to `cfd-core`.** Every array crossing a crate boundary is
//!   exactly `grid.len()` long, interior only. Ghost cell widths mirror the interior.
//! - **Ping-pong.** Read `u_old` immutably, write `u_new` through `par_chunks_mut` over
//!   rows. Never mutate in place.
//! - **One error type**, `CfdError`, defined here. No crate defines its own and none
//!   returns `anyhow::Result` in its public API.
//! - **`Prim` is rotated to the face.** `Prim = [rho, u_n, u_t, p]` where `n` is the sweep
//!   direction: z-sweep `u_n = u_z`, `u_t = u_r`; r-sweep `u_n = u_r`, `u_t = u_z`. `hllc_flux`
//!   returns a `Cons` in that same rotated frame; the caller rotates back. `compute_primitives`
//!   produces the **canonical unrotated** `[rho, u_z, u_r, p]`; the sweeps rotate at use.

pub mod kernels;

use std::sync::Arc;

pub type Real = f32;

pub type Cons = [Real; 4]; // [rho, rho*u_z, rho*u_r, E]
pub type Prim = [Real; 4]; // [rho, u_n, u_t, p] — rotated to the sweep direction. See conventions.

pub const NG: usize = 2; // ghost width. MUSCL needs 2. Baked into the padded layout.
pub const SOLID_THRESHOLD: f32 = 0.5; // cells at/above this fraction are frozen, never updated
pub const CFL_DEFAULT: Real = 0.4;
// Non-dimensional floors. See docs/physics-reference.md §5. The pressure
// floor is 1e-6 of chamber pressure so it tracks the 8.6–300 bar range of
// the real-engine presets; the UI's 58 km altitude cap and fixed-back-
// pressure vacuum mode keep ambient strictly above it.
pub const RHO_MIN: Real = 1e-8;
pub const P_MIN_ABS: Real = 1e-6;

#[derive(Debug, thiserror::Error)]
pub enum CfdError {
    #[error("invalid geometry: {0}")] Geometry(String),
    #[error("invalid grid: {0}")] Grid(String),
    #[error("invalid parameter: {0}")] Parameter(String),
    #[error("solver diverged at step {step}: {detail}")] Diverged { step: u64, detail: String },
    #[error("io: {0}")] Io(#[from] std::io::Error),
}
pub type Result<T> = std::result::Result<T, CfdError>;

/// Tensor-product, anisotropic, axisymmetric (z, r) grid of cell EDGES —
/// graded or uniform, the solver cannot tell. All lengths non-dimensional, in
/// units of throat radius. `z_edges[0] == 0` and `r_edges[0] == 0` (the axis)
/// always. Cheap to clone: the geometry arrays live behind one `Arc`.
///
/// Per-cell geometry comes precomputed, padded to the ghost layout (ghost
/// widths mirror the interior; radial ghost positions mirror across the
/// axis). `_g` accessors take padded (possibly negative) indices; the plain
/// ones take interior indices.
#[derive(Debug, Clone)]
pub struct Grid {
    pub nz: usize,
    pub nr: usize,
    geom: Arc<AxisGeom>,
}

#[derive(Debug)]
struct AxisGeom {
    z_edges: Vec<f64>, // nz + 1, strictly increasing, [0] == 0
    r_edges: Vec<f64>, // nr + 1, strictly increasing, [0] == 0
    zw: Vec<Real>,     // snz padded cell widths
    rw: Vec<Real>,     // snr padded cell widths
    zc: Vec<Real>,     // snz padded cell centres (midpoints)
    rc: Vec<Real>,     // snr padded cell centres (arithmetic mean of face radii)
    rcv: Vec<Real>,    // snr padded radial VOLUME centroids (mirrored below the axis)
    dz_min: Real,
    dr_min: Real,
    uniform: bool,
}

impl PartialEq for Grid {
    fn eq(&self, o: &Self) -> bool {
        self.nz == o.nz
            && self.nr == o.nr
            && (Arc::ptr_eq(&self.geom, &o.geom)
                || (self.geom.z_edges == o.geom.z_edges && self.geom.r_edges == o.geom.r_edges))
    }
}

/// Padded widths, centres and (for r) volume centroids for one axis.
/// Ghost widths mirror the interior; positions continue outward from the
/// interior centres. `mirror_low` reflects the low-side ghost positions
/// across x = 0 exactly (the axis), which the r axis needs so the mirror
/// ghost rows sit at the mirrored radii bit-exactly.
fn pad_axis(edges: &[f64], mirror_low: bool) -> (Vec<Real>, Vec<Real>, Vec<Real>) {
    let n = edges.len() - 1;
    let sn = n + 2 * NG;
    let mut w = vec![0.0 as Real; sn];
    let mut c = vec![0.0 as Real; sn];
    let mut cv = vec![0.0 as Real; sn];
    // Interior.
    for i in 0..n {
        let (lo, hi) = (edges[i], edges[i + 1]);
        w[NG + i] = (hi - lo) as Real;
        c[NG + i] = (0.5 * (lo + hi)) as Real;
        // Volume centroid of the shell [lo, hi]: (2/3)(hi^3-lo^3)/(hi^2-lo^2).
        cv[NG + i] = ((2.0 / 3.0) * (hi * hi + hi * lo + lo * lo) / (hi + lo)) as Real;
    }
    // Ghost widths mirror the interior (clamped for n < NG).
    for k in 1..=NG {
        w[NG - k] = w[NG + (k - 1).min(n - 1)];
        w[NG + n - 1 + k] = w[NG + n - k.min(n)];
    }
    // Low-side ghost positions.
    if mirror_low {
        for k in 1..=NG {
            c[NG - k] = -c[NG + (k - 1).min(n - 1)];
            cv[NG - k] = -cv[NG + (k - 1).min(n - 1)];
        }
    } else {
        for k in 1..=NG {
            c[NG - k] = c[NG - k + 1] - 0.5 * (w[NG - k] + w[NG - k + 1]);
            cv[NG - k] = c[NG - k];
        }
    }
    // High-side ghost positions: step outward; centroids from extended edges.
    let mut e_hi = edges[n];
    let mut prev_c = c[NG + n - 1];
    for k in 1..=NG {
        let wid = w[NG + n - 1 + k];
        c[NG + n - 1 + k] = prev_c + 0.5 * (w[NG + n - 2 + k] + wid);
        prev_c = c[NG + n - 1 + k];
        let (lo, hi) = (e_hi, e_hi + wid as f64);
        cv[NG + n - 1 + k] = ((2.0 / 3.0) * (hi * hi + hi * lo + lo * lo) / (hi + lo)) as Real;
        e_hi = hi;
    }
    (w, c, cv)
}

impl Grid {
    /// Uniform grid — the historical special case. Panics on non-positive or
    /// non-finite spacing: uniform construction is programmer input, not user
    /// input (user-supplied edges go through `from_edges`, which returns Err).
    pub fn uniform(nz: usize, nr: usize, dz: Real, dr: Real) -> Grid {
        assert!(nz > 0 && nr > 0, "Grid::uniform: nz={nz} nr={nr}");
        assert!(
            dz.is_finite() && dz > 0.0 && dr.is_finite() && dr > 0.0,
            "Grid::uniform: dz={dz} dr={dr}"
        );
        let z_edges: Vec<f64> = (0..=nz).map(|i| i as f64 * dz as f64).collect();
        let r_edges: Vec<f64> = (0..=nr).map(|i| i as f64 * dr as f64).collect();
        Self::build(z_edges, r_edges, true)
    }

    /// Arbitrary cell edges, one list per axis. Each list needs at least 2
    /// entries, must start at exactly 0 and be strictly increasing and finite.
    pub fn from_edges(z_edges: Vec<f64>, r_edges: Vec<f64>) -> Result<Grid> {
        for (name, e) in [("z", &z_edges), ("r", &r_edges)] {
            if e.len() < 2 {
                return Err(CfdError::Grid(format!("{name}_edges has {} entries", e.len())));
            }
            if e[0] != 0.0 {
                return Err(CfdError::Grid(format!("{name}_edges[0] = {} (must be 0)", e[0])));
            }
            for i in 1..e.len() {
                if !(e[i].is_finite() && e[i] > e[i - 1]) {
                    return Err(CfdError::Grid(format!(
                        "{name}_edges not strictly increasing at {i}: {} -> {}",
                        e[i - 1], e[i]
                    )));
                }
            }
        }
        Ok(Self::build(z_edges, r_edges, false))
    }

    fn build(z_edges: Vec<f64>, r_edges: Vec<f64>, uniform: bool) -> Grid {
        let nz = z_edges.len() - 1;
        let nr = r_edges.len() - 1;
        let (zw, zc, _) = pad_axis(&z_edges, false);
        let (rw, rc, rcv) = pad_axis(&r_edges, true);
        let dz_min = zw[NG..NG + nz].iter().cloned().fold(Real::INFINITY, Real::min);
        let dr_min = rw[NG..NG + nr].iter().cloned().fold(Real::INFINITY, Real::min);
        Grid {
            nz,
            nr,
            geom: Arc::new(AxisGeom { z_edges, r_edges, zw, rw, zc, rc, rcv, dz_min, dr_min, uniform }),
        }
    }

    /// Interior cell count. Every slice crossing a crate boundary is this long.
    pub fn len(&self) -> usize { self.nz * self.nr }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn idx(&self, iz: usize, ir: usize) -> usize { ir * self.nz + iz } // row-major, z contiguous
    pub fn snz(&self) -> usize { self.nz + 2 * NG } // padded row stride, ghosts both sides
    pub fn snr(&self) -> usize { self.nr + 2 * NG }
    pub fn glen(&self) -> usize { self.snz() * self.snr() }
    /// Padded index. Accepts negative interior coordinates for ghost access.
    pub fn gidx(&self, iz: isize, ir: isize) -> usize {
        ((ir + NG as isize) as usize) * self.snz() + ((iz + NG as isize) as usize)
    }

    /// Cell edges (faces), f64, `nz + 1` / `nr + 1` long. The source of truth
    /// the per-cell caches derive from; the rasterizer clips against these.
    pub fn z_edges(&self) -> &[f64] { &self.geom.z_edges }
    pub fn r_edges(&self) -> &[f64] { &self.geom.r_edges }

    // ---- per-cell geometry, interior indices ------------------------------
    pub fn dz(&self, iz: usize) -> Real { self.geom.zw[NG + iz] }
    pub fn dr(&self, ir: usize) -> Real { self.geom.rw[NG + ir] }
    /// Finest spacing on each axis — what the timestep and display resolution
    /// key on. NOT a substitute for the per-cell width in any flux formula.
    pub fn dz_min(&self) -> Real { self.geom.dz_min }
    pub fn dr_min(&self) -> Real { self.geom.dr_min }
    pub fn z_face(&self, iz: usize) -> Real { self.geom.z_edges[iz] as Real }
    /// Lower face radius of row ir. Row 0 returns exactly 0.
    pub fn r_face(&self, ir: usize) -> Real { self.geom.r_edges[ir] as Real }
    pub fn z_center(&self, iz: usize) -> Real { self.geom.zc[NG + iz] }
    /// Cell-centre radius: the ARITHMETIC MEAN of the two face radii (exact
    /// volume radius — `r_center*dr` is the true shell area). Row 0 returns
    /// dr(0)/2, never 0. Reconstruction must NOT use this — see `r_centroid_g`.
    pub fn r_center(&self, ir: usize) -> Real { self.geom.rc[NG + ir] }

    // ---- per-cell geometry, padded indices (kernel stencils reach ghosts) --
    pub fn dz_g(&self, iz: isize) -> Real { self.geom.zw[(iz + NG as isize) as usize] }
    pub fn dr_g(&self, ir: isize) -> Real { self.geom.rw[(ir + NG as isize) as usize] }
    pub fn z_center_g(&self, iz: isize) -> Real { self.geom.zc[(iz + NG as isize) as usize] }
    pub fn r_center_g(&self, ir: isize) -> Real { self.geom.rc[(ir + NG as isize) as usize] }
    /// Radial VOLUME centroid of the shell, (2/3)(r_hi³-r_lo³)/(r_hi²-r_lo²),
    /// mirrored (negative) in the below-axis ghost rows. This is where a cell
    /// average of a linear-in-r field actually sits; radial reconstruction in
    /// axisymmetric mode uses these positions. It differs from `r_center` by
    /// up to dr/6 at the axis (~4% for graded rows) — do not mix them up.
    pub fn r_centroid_g(&self, ir: isize) -> Real { self.geom.rcv[(ir + NG as isize) as usize] }

    /// Cell volume / (2*pi): r_center*dr*dz in f64 from the exact edges.
    pub fn cell_vol(&self, iz: usize, ir: usize) -> f64 {
        let (rl, rh) = (self.geom.r_edges[ir], self.geom.r_edges[ir + 1]);
        let (zl, zh) = (self.geom.z_edges[iz], self.geom.z_edges[iz + 1]);
        0.5 * (rl + rh) * (rh - rl) * (zh - zl)
    }

    /// Domain extents.
    pub fn lz(&self) -> f64 { *self.geom.z_edges.last().unwrap() }
    pub fn lr(&self) -> f64 { *self.geom.r_edges.last().unwrap() }

    /// Interior cell containing coordinate z (clamped into range).
    pub fn z_cell_at(&self, z: f64) -> usize {
        self.geom.z_edges[1..].partition_point(|&e| e <= z).min(self.nz - 1)
    }
    pub fn r_cell_at(&self, r: f64) -> usize {
        self.geom.r_edges[1..].partition_point(|&e| e <= r).min(self.nr - 1)
    }

    /// True for grids built by `Grid::uniform`. Display fast paths only —
    /// no numerics may branch on this.
    pub fn is_uniform(&self) -> bool { self.geom.uniform }
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
    pub fn empty(grid: Grid) -> Self {
        let n = grid.len();
        SolidField { grid, fraction: vec![0.0; n] }
    }
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

#[cfg(test)]
mod grid_tests {
    use super::*;

    #[test]
    fn uniform_grid_matches_legacy_geometry() {
        let g = Grid::uniform(320, 200, 0.1449, 0.05);
        assert_eq!(g.len(), 64_000);
        assert_eq!(g.idx(3, 2), 2 * 320 + 3);
        assert_eq!(g.gidx(-2, -2), 0);
        for ir in 0..g.nr {
            assert!((g.r_center(ir) - (ir as Real + 0.5) * 0.05).abs() <= 1e-5);
            assert!((g.r_face(ir) - ir as Real * 0.05).abs() <= 1e-5);
            assert!((g.dr(ir) - 0.05).abs() <= 1e-7);
        }
        assert_eq!(g.r_face(0), 0.0);
        assert!((g.dz_min() - 0.1449).abs() <= 1e-7);
        assert!((g.cell_vol(0, 0) - 0.025 * 0.05 * 0.1449).abs() < 1e-9);
        assert!(g.is_uniform());
        assert_eq!(g.clone(), g);
    }

    #[test]
    fn graded_grid_geometry_is_consistent() {
        let g = Grid::from_edges(
            vec![0.0, 1.0, 2.0, 3.5, 5.5],
            vec![0.0, 0.5, 1.0, 1.8, 3.0],
        )
        .unwrap();
        assert_eq!((g.nz, g.nr), (4, 4));
        assert_eq!(g.dz(2), 1.5);
        assert_eq!(g.dr(3), 1.2);
        assert_eq!(g.lz(), 5.5);
        assert_eq!(g.lr(), 3.0);
        // r_center is the arithmetic face mean; r_centroid sits above it.
        assert_eq!(g.r_center(2), 1.4);
        let cv = g.r_centroid_g(2) as f64;
        let exact = (2.0 / 3.0) * (1.8f64.powi(3) - 1.0) / (1.8f64.powi(2) - 1.0);
        assert!((cv - exact).abs() < 1e-6);
        assert!(cv > 1.4);
        // Ghost widths mirror; below-axis positions mirror exactly.
        assert_eq!(g.dr_g(-1), g.dr(0));
        assert_eq!(g.dr_g(-2), g.dr(1));
        assert_eq!(g.r_center_g(-1), -g.r_center(0));
        assert_eq!(g.r_centroid_g(-2), -g.r_centroid_g(1));
        assert_eq!(g.dz_g(4), g.dz(3));
        // Coordinate lookup.
        assert_eq!(g.z_cell_at(0.2), 0);
        assert_eq!(g.z_cell_at(3.4), 2);
        assert_eq!(g.z_cell_at(99.0), 3);
        assert_eq!(g.r_cell_at(-1.0), 0);
        // Volume matches the exact shell integral.
        let vol = g.cell_vol(1, 2);
        assert!((vol - 0.5 * (1.0 + 1.8) * 0.8 * 1.0).abs() < 1e-12);
        assert!(!g.is_uniform());
    }

    #[test]
    fn from_edges_rejects_bad_input() {
        assert!(Grid::from_edges(vec![0.0], vec![0.0, 1.0]).is_err());
        assert!(Grid::from_edges(vec![0.5, 1.0], vec![0.0, 1.0]).is_err());
        assert!(Grid::from_edges(vec![0.0, 1.0, 1.0], vec![0.0, 1.0]).is_err());
        assert!(Grid::from_edges(vec![0.0, 1.0], vec![0.0, f64::NAN]).is_err());
        assert!(Grid::from_edges(vec![0.0, 1.0, 2.0], vec![0.0, 1.0]).is_ok());
    }
}
