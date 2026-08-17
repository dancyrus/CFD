# Work order: non-nozzle silent-wrong-answer guards (A2)

**Status:** landed. Three fixes that remove the worst silent wrong-answer
paths for non-nozzle geometry, one guard, and the measurement that mattered
more than any of them: how much of G0's and G3's A1 failures was the outflow
boundary rather than the staircase wall. Answer below, unrounded.

All "measured" figures for the defects were taken against the pre-fix code on
non-nozzle geometry — a 3×3 array of disconnected disks, and a duct with thin
baffles. None of these paths is exercised by a nozzle, which is why eighteen
green T rungs never saw any of them.

## The measurement that matters: was A1's G0/G3 failure the boundary?

A1's rungs G0–G3 failed 7 assertions and the write-up attributed all of them
to the immersed staircase wall. G1 and G2 are clean supersonic outflow, so the
outflow boundary cannot touch them. But G0 (M 0.3 ellipse) and G3
(forward-facing step) both have recirculation that could reach a boundary, and
fix 1 changes exactly that boundary — so before anyone commits weeks to a
cut-cell rewrite, this session instrumented first, fixed second, and measured
the difference.

### a. Instrumentation, BEFORE any fix

An outflow-plane probe (now permanent in the ladder, recorded to
`docs/results/` before any assert) tracks over each run: the minimum `u_z` in
the boundary-adjacent cells, the peak fraction of outflow-boundary cells with
`u_z < 0` at one instant, and the peak fraction with `u_z ≤ −a` (the branch
the pre-fix code misread as supersonic *outflow*).

| case | min u_z at outflow plane | peak fraction reversed | peak fraction u_z ≤ −a |
|---|---|---|---|
| G0 h = 0.1 | −1.381e-1 | 0.154 | 0 |
| G0 h = 0.05 | −6.850e-1 | 0.600 | 0 |
| G0 h = 0.025 | −4.953e-1 | 0.600 | 0 |
| G3 | **+2.100e0** | **0** | 0 |

Reversed flow reaches G0's outflow plane massively — at the finer meshes more
than half the exit column is inflowing at some instant, at up to 65% of the
freestream speed. It **never** reaches G3's outflow plane, so fix 1 is not
implicated in G3 and its momentum failure keeps its A1 diagnosis
(trapezoid quadrature of the wall impulse, second order in dt).

### b. Fix 1 ALONE (fixes 2 and 3 not yet applied)

G0 at h = 0.1 / 0.05 / 0.025, G3 at h = 1/80, same rigs as A1:

| quantity | before | after fix 1 alone |
|---|---|---|
| G0-steady (settled) | true / false / false | **true / true / true** |
| G0 residual | 1.59e-4 / 5.75e-3 / 5.48e-3 | 5.81e-5 / 6.74e-5 / 2.08e-4 |
| G0 peak-to-peak C_d | 1.20e-2 / 6.07e-1 / 1.01e0 | **3.41e-6 / 3.48e-4 / 9.53e-3** |
| G0 C_d (exact 0) | 7.132e-1 / 6.453e-1 / 5.247e-1 | 7.181e-1 / 7.056e-1 / 6.661e-1 |
| G0-order (fine pair, pass ≥ 1.0) | 0.298 | 0.083 |
| G0 max Mach | 0.303 / 0.348 / 0.428 | 0.304 / 0.305 / 0.337 |
| G0 min u_z at the plane | −0.138 / −0.685 / −0.495 | −0.016 / −0.033 / −0.025 |
| G3-momentum (tol 2e-6) | 5.388e-5 | **5.388e-5 — bit-identical** |
| G3 mass / energy | 4.298e-7 / 1.231e-7 | 4.298e-7 / 1.231e-7 — bit-identical |

### What the table decides

**The "unsteady inviscid wake" was the boundary, not the staircase.** A1's
most alarming G0 finding — a solver with no viscosity producing an oscillating
separated wake that got *worse* under refinement — was the old outflow ghost
feeding the recirculation: every reversed cell at the exit plane extrapolated
*interior* entropy and tangential velocity along characteristics that were
entering the domain, a self-exciting loop. With the reversal handled, the flow
settles at **every** level by the project's own §9 definition, and the C_d
oscillation collapses by three to five orders of magnitude. G0-steady now
passes; so does the monotone-decrease gate on G0-cd.

**The drag magnitude and its decay rate are the staircase, and that diagnosis
stands.** C_d ≈ 0.67 against an exact 0 at h = 0.025, decaying at order 0.083
— *further* below first order than the oscillating pre-fix numbers suggested,
now that the number is actually a measurement (steady, spread 9.5e-3 around a
mean of 0.666). G0-order still fails, and it fails cleanly now. G1, G2 are
untouched by construction (verified: identical failures), G3 is untouched to
the bit. The staircase work — and the cut-cell decision it feeds — should be
sized against **this** table, not A1's: the staircase's G0 signature is a
large, slowly-decaying steady drag bias, not unsteadiness.

**One caution on reading G0-order:** the pre-fix 0.298 was the log-ratio of
two numbers riding an oscillation larger than themselves; the post-fix 0.083
is the log-ratio of two converged means. They are not comparable "before vs
after" values of the same quantity — only the post-fix number is real.

## The fixes

### 1. `outflow_ghost` — reversed-flow branch, signed supersonic test

`cfd-core/src/physics.rs`. The downstream outflow ghost had no reversed-flow
branch, and its supersonic test was `|u| ≥ a`. Two silent failure modes:

- Reversed subsonic flow fell into the subsonic-outflow construction, which
  extrapolates interior entropy and tangential velocity along incoming
  characteristics. Measured: interior u = −0.30 → ghost u = +1.45, a spurious
  outward jet at M 2.5.
- Supersonic *inflow* (u ≤ −a) satisfied `|u| ≥ a` and returned a bit-exact
  copy of the interior — zero conditions imposed on four incoming
  characteristics, ill-posed. Measured: a sustained M 4.49 inflow grew domain
  mass 41.7% over 3000 steps with nothing constraining it.

Now the same three-way split `farfield_ghost` already had on the radial
normal: `u < 0` → ambient reservoir at rest (incoming gas carries ambient
entropy and ambient tangential velocity, never the interior's); `0 ≤ u < a` →
impose p_a, extrapolate entropy and R⁺; `u ≥ a` → copy. Reference §4 updated.

### 2. `quasi1d_init` — the nozzle gate, extracted and tightened

The gate required only a lip, an interior argmin of open radius, and the
argmin smaller than both ends — satisfied by a blockage anywhere. Measured: it
seeded 73.1% of the disk array's fluid at up to M 2.07, and 98.4% of the
baffled duct at up to M 3.13 and 0.996 p₀ against 0.0203 p₀ ambient — a 50×
overpressure blast wave at t = 0, on by default, in cases with no nozzle.

The predicate is now `physics::nozzle_profile`, a named public function —
**work order A3 needs the identical predicate and must call this one, not
restate it.** It additionally requires exactly one fluid run per column (inlet
through lip) and a converge-then-diverge open-radius profile. On gate failure
the fallback is generic: uniform isentropic freestream when the
chamber-to-ambient expansion is supersonic *and* the ambient lies on the
chamber's isentrope (the external-flow rig); ambient at rest otherwise. The
isentrope condition is what stops a rocket chamber over a 50× pressure ratio
from initializing M 3.2 uniform flow through arbitrary drawn geometry — the
same blast wave the gate exists to prevent. Each path logs one line;
after a few thousand steps the init is invisible and the log is the only
record of which ran. Reference §5 updated.

### 3. `apply_geometry_change` — BFS to completion, extrapolated cavities, momentum on the ledger

Three defects in one function:

- The refill BFS capped at 8 passes. Fine for nudging a wall; wrong for
  erasing an obstacle, which is what a sandbox user does. Measured cells left
  unreached: 6×6 block 0%, 16×16 0%, **40×40 48%, 80×80 80%**. It now runs to
  completion — the loop terminates naturally (each pass fills a cell or the
  frontier is empty).
- Unreached cells were dumped at ambient. A sealed cavity inside a hot body
  filled with cold far-field ambient is a spurious pressure discontinuity.
  Now: the (ρ, p) of the *nearest valid fluid cell* by grid distance, at
  rest, via a multi-source BFS from all valid cells through solid; ambient
  only when the edit leaves no valid fluid anywhere.
- `FlipLedger` tracked only mass and energy, so a large edit silently
  injected/removed **momentum** with no audit trail — and `Δtotal == ledger`,
  the invariant that catches exactly this, was not defined for momentum. The
  ledger now books `momentum_z`/`momentum_r`: closures remove all four
  conserved totals, refills are at rest and book zero. Reference §3 and
  contract.md updated (contract change: both docs mirrored in this series,
  full ladder re-run).

### Guard: `WallMode::ColumnReflect` refused off star-convex geometry

`column_reflect_fill` finds the first solid cell from the axis and overwrites
everything above it. On a column with a second fluid run that destroys live
fluid: measured on the 9-disk array, **2346 of 21420 live fluid cells
overwritten, report off by 2.6×**, silently. On the nozzle: 0 of 15977 —
why no rung ever saw it. `physics::column_reflect_supported` (at most one
solid run per column) now gates the mode and `step()` returns
`CfdError::Geometry` instead of running it. Refusing loudly is the correct
behaviour for an abort-plan flag: the mode exists as a fallback for nozzles,
and a fallback that silently corrupts the case it falls back on is worse than
no fallback.

## Verification

- T0–T7 and the graded-grid guards: all pass, **bit-identical values** —
  T2 drift 2.982e-7/8.342e-8, T3 orders 1.988/1.776, T5 floors 0 with the
  same sponge-entry pressure to the digit, T6 beta 46.448, T7 beta 27.961,
  T1/T4 graded to every printed digit.
- **T8 moved, and that is a finding, reported as the work order requires.**
  N20: mdot/ideal 0.9032 → 0.9035, area-avg M 2.085 → 2.086, delta50
  unchanged at 0.134 r_t. N40: mdot/ideal 0.9262 → 0.9358, M 2.258 → 2.214,
  delta50 0.079 → 0.082 r_t; the asserted 40/20 thickness ratio went
  0.586 → 0.614, inside its 0.35–0.70 band, floors 0/0. The premise "the
  nozzle exercises none of these paths" is not quite exact: the T8 rig's
  *downstream boundary* spans both the supersonic core (untouched) and the
  subsonic ambient region at outer radii, where startup entrainment toward
  the jet transiently reverses u_z at the exit plane. Fix 1 changes exactly
  that state — entrained inflow now carries ambient entropy instead of
  extrapolated interior entropy — so a small movement in the plume-adjacent
  measurements at the finer nozzle mesh is the fix working, not a regression.
  T2/T5/T6/T7 have closed or supersonic boundaries and did not move at all.
- Unit suite: 48 pass, including new tests for the reversed-outflow branches,
  the disk-array/baffle/floating-baffle gate rejections, the CD-nozzle gate
  acceptance, the isentrope-gated freestream fallback, the 40×40
  full-refill + four-way ledger invariant, the sealed-cavity extrapolation,
  and the ColumnReflect run counter.
- G rungs after all fixes (recorded in `docs/results/ladder-intel-xeon-4c.json`):
  - **G0**: floors/shock-free pass, **steady passes** (new), C_d decrease
    passes, order still fails (0.083 < 1.0) — the staircase's number.
  - **G1**: unchanged — fails p₃/p₁ and reflection point exactly as A1
    measured (clean supersonic outflow; fix 1 cannot and does not touch it).
  - **G2**: unchanged — fails C_d at 341%, same mechanism.
  - **G3**: mass/energy/floors/stem pass, momentum fails at 5.388e-5,
    bit-identical to A1 — quadrature-dominated, fix lives in `step()`
    accumulating the wall impulse across both RK stages (out of A2's scope).

## Reproducing

```
cargo test -p cfd-core                                            # unit suite
cargo test -p cfd-core --test ladder -- --include-ignored --nocapture
cargo test -p cfd-core --test acceptance -- --include-ignored --nocapture
```

The outflow-plane probe lines print with the G0/G3 output and are recorded as
notes in the ladder results JSON.
