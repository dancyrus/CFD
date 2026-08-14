# CFD Sandbox — agent instructions

2D axisymmetric compressible Euler sandbox. Rust workspace, four crates, built by
parallel sessions each owning one file or crate.

**Read `docs/physics-reference.md` before touching numerics.** It is the authority
on every formula, constant and tolerance. Do not deviate from it. If you think it
is wrong, say so and stop.

**Read `docs/contract.md` before writing any cross-crate code.** It holds the exact
type and function signatures every crate codes against.

**Find your session brief in `docs/sessions/` and follow it.** If you do not know
which session you are, ask.

## Crate ownership — do not cross these lines

| Path | Owner |
|---|---|
| `cfd-contract/` | coordinator — changeable, see the contract-change rule below |
| `cfd-core/src/lib.rs`, `step.rs` | coordinator — changeable, see the contract-change rule below |
| `cfd-core/src/kernel.rs` | session A — sweeps, dt, rayon, positivity |
| `cfd-core/src/physics.rs` | session B — axisym source, wall, BCs, sponge, init |
| `cfd-geom/` | session C — contours, rasterizer, editor data model. No egui. |
| `cfd-ui/` | session D — the app. **Sole owner of eframe/egui.** |

**Contract changes.** The freeze on `cfd-contract/`, `cfd-core/src/lib.rs` and
`step.rs` applied during the parallel build phase, which is over. Changes are
now allowed, under two conditions, both non-negotiable:

1. Mirror the change into `docs/contract.md` and `docs/physics-reference.md`
   **in the same commit** — the docs are the authority and must never disagree
   with the code.
2. Re-run the **full acceptance ladder** (`cargo test -p cfd-core --test ladder
   -- --include-ignored` plus `--test acceptance -- --include-ignored`) and do
   not commit unless it is green.

Precedent: the `P_MIN_ABS` raise from 1e-8 to 1e-6 (real-engine presets) did
exactly this — constant, both docs, and T1–T8 re-run with floors = 0.

## Conventions that cause silent bugs if broken

- Row-major, z contiguous: `idx = ir*nz + iz`. Use `Grid::idx`, never open-code it.
- The grid is a tensor product of arbitrary cell-edge lists (graded or uniform —
  see docs/work-orders/grid-grading.md). The lower face of row 0 is at `r = 0`;
  `r_center` (arithmetic mean of face radii) is never zero. Reconstruction uses
  `r_centroid_g`; the p/r balance uses face pressures and no cell radius at all.
  Never index-scale a coordinate (`i*dz`): go through `Grid`'s accessors.
- Ghost cells are private to `cfd-core`. Every array crossing a crate boundary is
  exactly `grid.len()` long, interior only.
- All solver state is non-dimensional, chamber-referenced. `R = 1`, `p = rho*T`,
  chamber is exactly `(1, 0, 0, 1)`. SI appears only in `Snapshot` and `Report`.
- `f32` in the hot loop (`Real`). **Every reduction accumulates in `f64`** — mass,
  momentum, energy, thrust, L1 norms, residuals. An f32 sum over 64k cells has a
  worst case larger than the Sod pass threshold.
- `Prim` is `[rho, u_n, u_t, p]`, rotated to the sweep direction. `compute_primitives`
  produces the canonical unrotated `[rho, u_z, u_r, p]`; sweeps rotate at use.
- Ping-pong: read `u_old` immutably, write `u_new` through `par_chunks_mut` over
  rows. Never mutate in place.
- One error type, `CfdError`, in the contract. No crate defines its own. No
  `anyhow` in a public API.
- The radial flux difference and the axisymmetric pressure source go inside **one
  bracket**. The f32 cancellation depends on it.
- Image row 0 is the largest r, the opposite of `Grid`. Only `cfd-ui` flips.

## Results get committed, not reported in chat

Numbers that only live in a chat transcript are lost. The acceptance ladder
and every benchmark write their measured results to
`docs/results/<suite>-<machine>.json` through the `cfd-results` crate —
automatically, from inside each test **before its own asserts**, so emission
cannot be skipped, forgotten, or lost to a failure. The machine label is
derived from the hardware (CPU brand + core count), never from a flag, so the
same machine always writes the same file and results are diffable across
time. The schema is documented in `docs/results/README.md`; extend that
schema, never invent a second format. `docs/results/` must never be
gitignored. Commit the updated result files together with the change that
produced them.

## Testing

- `cargo test -p <crate>`. Never a bare workspace build — it serializes on the
  cargo build lock across worktrees.
- `[profile.dev.package."*"] opt-level = 3` is set. **Do not remove it.** An
  unoptimized MUSCL-HLLC kernel is 20–50× slower and reads as a performance bug.
- **Never fix a failing physics test by loosening a tolerance.** Report it.
- Acceptance ladder and its reference values: `docs/physics-reference.md` §12.

## Pinned versions — do not run `cargo update`

`eframe`/`egui` `=0.31.1` (0.34 deprecated `App::update` in favour of `App::ui`
and unified the panel types). `rayon` `=1.10.0`. `triple_buffer` `=8.0.0`.

`eframe` and `egui` are dependencies of `cfd-ui` **only**. Each worktree has its
own empty `target/` and eframe is a 2–5 minute cold build; keeping it in one crate
means one session pays instead of four. Do not add `wgpu` as a direct dependency
anywhere — the app renders one textured quad and gets the GPU through eframe.

## The five abort-plan flags

`Geometry {Axisymmetric, Planar}`, `Reconstruction {FirstOrder, Muscl}`,
`WallMode {Mirror, ColumnReflect}`, `SolverKind {Mock, Real}`,
`FluxMode {Hllc, HllRadial, Hll}`.

Every one is honoured by the solver from day one. They are the fallback plan and
they exist before there is any pressure to write them. Do not remove or stub them.

## What this build deliberately does not have

Cut cells. Body-fitted meshing. GPU compute. Viscosity, turbulence, heat transfer,
separation. Species transport or mixture gamma. AMR. Local timestepping. Case
save/load. Adding any of these is a scope change, not a fix.
