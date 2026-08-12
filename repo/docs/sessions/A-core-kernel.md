# Session A — `cfd-core/src/kernel.rs`

**Budget: 105 minutes. You are the critical path.**

## Read first

`docs/physics-reference.md`, `docs/contract.md`, `cfd-contract/src/lib.rs`, `cfd-contract/src/kernels.rs`. Re-read `cfd-contract/src/lib.rs` before every edit — your context will be compacted and the contract is what you must not drift from.

## Rules

- You own **`cfd-core/src/kernel.rs` and nothing else.**
- Never modify anything in `cfd-contract/`, or `cfd-core/src/lib.rs`, or `cfd-core/src/step.rs`. If you believe the contract is wrong, STOP and print `CONTRACT CHANGE REQUEST:` followed by the exact diff you want. Do not make the change.
- `cargo test -p cfd-core`, never a bare workspace build.
- All state is non-dimensional. Every reduction accumulates in `f64`.
- Ping-pong: read `u_old` immutably, write `u_new` through `par_chunks_mut` over rows. **Never mutate in place.** Doing otherwise is a guaranteed forty-minute fight with the borrow checker.

## What you build

Implement the `todo!()` signatures already declared in `kernel.rs`:

```rust
pub fn compute_primitives(u: &[Cons], w: &mut [Prim], gamma: Real);
pub fn max_wave_speed(w: &[Prim], solid: &[bool], g: &Grid, gamma: Real) -> Real;
pub fn sweep_z(...);
pub fn sweep_r(...);
pub fn accumulate(...);
pub fn accumulate2(...);
pub fn enforce_positivity(u: &mut [Cons], gas: &GasModel, floors: &mut u64);
pub fn density_residual_f64(u0: &[Cons], u1: &[Cons], solid: &[bool]) -> f64;
```

### Sweeps

MUSCL reconstruction via `kernels::muscl_face_states`, flux via `kernels::hllc_flux` or `hll_flux` depending on `Numerics::flux_mode` and the carbuncle mask. Rotate to the sweep direction on the way in and back on the way out — `Prim` is `[rho, u_n, u_t, p]`, and `compute_primitives` produces the canonical unrotated `[rho, u_z, u_r, p]`.

At a fluid/solid face, call `crate::physics::wall_flux_z` or `wall_flux_r`. Session B owns those. **Do not implement them.** They are already declared with `todo!()` bodies, so your code compiles today.

`sweep_r` carries the axisymmetric machinery. The update is:

```
rhs_r[j] = − [ (r_{j+1/2}·G_{j+1/2} − r_{j−1/2}·G_{j−1/2}) − p_j·dr ] / (dr · r_j)
```

**Write it inside that single bracket.** Algebraically it is identical to computing the flux difference and adding `+p_j/r_j` separately, but in f32 the separate form leaves a residue of order `1e-7 × 40p × dt` per step at row 0 (where `p/r = 2p/dr`), which accumulates into a faint axis artifact. Inside one bracket the identical operands cancel bit-exactly. This is also what makes the well-balanced test pass.

Under `Geometry::Planar`, drop the r-weighting and the source entirely. This is abort-ladder rung 1 and it must work from day one.

Row 0's lower face has `r_face(0) = 0`, so it carries zero flux by construction regardless of the ghost state. The axis is a wall because of the geometry, not because of a boundary condition.

### Timestep

```
w_max = max over fluid cells of [ (|u|+a)/dz + (|v|+a)/dr ]
dt    = cfl / w_max          (cfl = 0.4)
```

Recompute every step — a nozzle startup swings the max wave speed by 5×, so a fixed dt either diverges during startup or wastes 5× at steady state. Fold the reduction into the same rayon pass that computes primitives; it is one extra streaming pass and effectively free.

### Positivity — two levels, both required

```
face level: if either reconstructed state has rho <= RHO_MIN or p <= p_min,
            replace BOTH states with their cell averages for that face
cell level: after each RK stage, if rho[c] <= RHO_MIN or p[c] <= p_min,
            redo cell c with all four faces first-order;
            if still bad, clamp and increment floors
```

A one-sided face fallback makes the flux multivalued and breaks conservation, so it must be both sides. And face-level fallback alone does not stop the cell average going negative after the update, which is why the cell-level pass exists. First-order HLLC/HLL with Davis speeds is provably positivity-preserving at CFL ≤ 0.5 and you run at 0.4, so the redo is sound.

Without the cell-level pass, the vacuum end of the altitude slider trips the floor counter — and the product refuses to display any number while that counter is nonzero. You would ship an app that shows nothing.

`p_min = max(P_MIN_ABS, 1e-4 * ambient.p)`.

### Honour the flags

`Reconstruction::FirstOrder` and every `FluxMode` variant must work. They are the abort ladder. `FirstOrder` + `Hll` + CFL 0.2 is ugly, diffuse and extremely robust, and it is what the human falls back to at minute 155 if the plume is NaN.

## Done when

```
cargo test -p cfd-core -- --ignored t1_freestream t4_sod well_balanced
```

passes. T4 is the one that matters: a correct second-order scheme measures L1(ρ) between 2.4e-3 and 4.1e-3; a first-order scheme measures 1.32e-2. The 6.0e-3 threshold sits in the gap, so it will catch a solver that is first order and calls itself MUSCL.

## Do not implement

Axisymmetric boundary conditions, walls, inlets, outflow, sponge, initialization, the carbuncle sensor. All of those are session B's, all are already declared, and all compile today.

## Known traps

- Debug builds. `[profile.dev.package."*"] opt-level = 3` is already set; do not remove it.
- Forgetting to rotate `Prim` for the r-sweep. `u_n` is the radial velocity there, not the axial one.
- Accumulating the residual or any conservation sum in f32. An f32 sum over 64k cells has a worst case of 3.8e-3, larger than the T4 threshold itself.
- Reconstructing across a solid cell. `muscl_face_states` handles this if you pass the `solid` array honestly; a fluid cell touching a wall must end up piecewise-constant in the wall-normal direction. That is the entire stencil degradation and nothing else is needed.
