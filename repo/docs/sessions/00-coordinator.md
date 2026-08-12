# Session 0 — Coordinator

**Budget: 50 minutes. Blocking — nothing else starts until you finish.**

You are the coordinator for a four-session parallel build of a 2D axisymmetric compressible Euler CFD sandbox in Rust. Four other Claude Code sessions will work in git worktrees off this repo, each owning one file or crate. Your job is to build the scaffold they code against, then stay alive as the integrator.

## Read first

- `docs/physics-reference.md` in full. It is the authority on every formula, constant and tolerance. Do not deviate from it. If you think it is wrong, say so and stop.
- `docs/contract.md` in full. It contains the exact type and function signatures you are about to transcribe.

## Rules

- Write **no real solver physics**. Your job is the scaffold, the two pure kernels, the orchestration, the tests, and the mock.
- Do not touch `cfd-geom` or `cfd-ui` beyond creating empty stub crates.
- `cargo test -p cfd-contract` must pass before you stop.

## Tasks, in order

### 1. Workspace (8 min)

Cargo workspace with members `cfd-contract`, `cfd-core`, `cfd-geom`, `cfd-ui`. Pin exact versions in the workspace `Cargo.toml`:

```toml
eframe        = "=0.31.1"
egui          = "=0.31.1"
rayon         = "=1.10.0"
triple_buffer = "=8.0.0"
thiserror     = "2"
```

`eframe` and `egui` are dependencies of **`cfd-ui` only**. No other crate may depend on them — each worktree gets its own empty `target/`, and eframe's tree is a 2–5 minute cold build. Keeping it in one crate means only one session pays.

Add:

```toml
[profile.dev.package."*"]
opt-level = 3
```

An unoptimized MUSCL-HLLC kernel is 20–50× slower than an optimized one, and a session that tests in debug will conclude the solver is too slow and start optimizing a non-problem.

Start `cargo build` in the background now so dependencies compile while you work. Commit `Cargo.lock`.

### 2. `cfd-contract/src/lib.rs` (17 min)

Transcribe the type listing in `docs/contract.md`. **Keep it under 250 lines.** Later sessions have their context compacted and must be able to re-read it cheaply.

Put the conventions block from `docs/contract.md` into the crate-level docs verbatim. It is the thing sessions re-read.

The five abort-plan enums — `Geometry`, `Reconstruction`, `WallMode`, `SolverKind`, `FluxMode` — must exist now and be honoured by the solver from day one. They are the fallback plan, and flags you have to write under pressure at minute 150 are flags you do not get.

### 3. `cfd-contract/src/kernels.rs` (12 min)

`minmod`, `cons_to_prim`, `prim_to_cons`, `sound_speed`, `hllc_flux`, `hll_flux`, `muscl_face_states`. Signatures in `docs/contract.md`.

These are the two most standard functions in numerical CFD and you should be able to write them near-verbatim. They live here, not in the solver crate, for a specific reason: it converts the seam between the two solver sessions from an agreement about behaviour into a shared compiled artifact. Contracts by signature drift at merge. Contracts by implementation cannot.

Tests that must pass:

- `hllc_flux` across the Sod interface reproduces the exact star state `p* = 0.3031301781`, `u* = 0.9274526200`.
- `muscl_face_states` on a linear field reproduces that field exactly.
- `prim_to_cons(cons_to_prim(u))` round-trips to 1e-6 relative.

### 4. `cfd-core/src/step.rs` and `lib.rs` (8 min)

`step.rs` is the complete SSP-RK2 orchestration, following the pseudocode in `docs/contract.md`. It calls functions in `kernel.rs` and `physics.rs` that do not exist yet — declare their exact signatures in those two files with `todo!()` bodies.

`lib.rs` is `mod kernel; mod physics; mod step;` plus re-exports.

**Both files are frozen after you commit.** Sessions A and B implement one file each and never open these. That is what makes the merge textually conflict-free.

The radial flux difference and the axisymmetric pressure source go inside a **single bracket**, exactly as `docs/physics-reference.md` §1 specifies. In f32 the separate form leaves a residue that accumulates into a faint axis artifact. Since `step.rs` is frozen, getting this right now is the only chance.

### 5. `cfd-core/tests/acceptance.rs` (skeleton) (3 min)

Three `#[test]`s, all `#[ignore]`d for now, written against the `Solver` trait:

- **T1** free-stream preservation.
- **T4** Sod on a 1-cell-tall strip, L1(ρ) ≤ 6.0e-3 at N = 200, t = 0.2, γ = 1.4. Use the exact reference values from `docs/physics-reference.md` §12.
- **Well-balanced** — uniform p, axisymmetric source cancels the radial flux difference to machine zero.

T4 requires the solver to support a planar 1-cell-tall mode. Make sure `Geometry::Planar` and the grid types allow it. Discovering that at minute 120 is a refactor.

Writing the tests before the solver exists gives session A a red-to-green target instead of letting it invent its own definition of done.

### 6. `MockSolver` (5 min)

In `cfd-core`, implementing `Solver`. An analytic expanding-cone plume: quasi-1D isentropic inside the nozzle, a Prandtl–Meyer fan at the lip, periodic shock cells, a Mach disk. It must produce a plausible **moving** `Snapshot` instantly.

Two reasons this is in the blocking phase rather than cut. It lets sessions C and D run end-to-end from minute one instead of waiting for the solver. And it is the bottom rung of the abort ladder — if the real kernel is broken at minute 150, this ships. It has to look convincing.

## Done when

`cargo test -p cfd-contract` is green and everything above is committed.

## After the fan-out

You stay alive as the integrator. You are the only session holding the contract in context and the only one that never needs re-briefing. See `docs/sessions/E-integration.md`.

While the four sessions run, your only job is contract changes. When a session prints `CONTRACT CHANGE REQUEST`, the human decides; if approved, the human pastes the identical change into every session. A session that edits the contract in its own worktree creates a merge conflict in the one file that must never conflict.
