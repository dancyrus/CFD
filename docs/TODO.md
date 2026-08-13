# TODO — next build

Ordered. Item 1 is the headline; everything below it is carried over from
`docs/build-plan.md` §7.

## 1. Cut cells (top priority)

Replace the sharp-threshold staircase wall with a cut-cell treatment of the
solid fraction the rasterizer already computes exactly.

**Justification — measured, not speculative** (integration, 2026-08-13;
diagnostics in `cfd-core/tests/diag.rs`, converted T8 in
`cfd-core/tests/ladder.rs`): a sloped wall on this Cartesian grid is a
staircase of axial faces, and each step sheds entropy. On the demo nozzle at
the default grid (N_throat = 20) the result is a **~12-radial-cell low-Mach
wall layer** — Mach falls from ~3.0 to ~0 across it at the exit plane — that
covers ~45% of the exit *area*. It costs ≈13% of thrust/C_f, ≈9% of mass
flow, and drags the area-averaged exit Mach ≈19% below the 1-D ideal (the
core flow is within ~7%). It was chased to ground during integration:

- The carbuncle sensor was firing on the smooth expansion (1,307 cells on a
  shock-free field) and was fixed to gate on compression — recovered only a
  few points.
- `WallMode::ColumnReflect` (radial-face reflection instead of the wall-flux
  formula) left the layer intact — it is the staircase geometry itself, not
  the wall-flux formulation.
- Refining dz alone (0.1449 → 0.10, §8's first lever) did not move it — the
  layer's thickness scales with **dr**.

Grid convergence is proven (T8 asserts the layer's physical thickness roughly
halves from N_throat 20 → 40), so the artifact is first-order in the wall
treatment: exactly what cut cells fix. Until then the T8 quasi-1D comparison
is a recorded measurement, and the app badges thrust and exit Mach with the
measured bias.

Note: `docs/deferred.md`-style reasoning from the original plan cut body-fitted
meshing, not cut cells on the existing rasterized fractions — the exact
sub-cell areas are already computed by `cfd-geom::rasterize`; the solver just
thresholds them at 0.5 today.

## 2. Drawn-geometry acceptance test

T0–T8 cover the parametric path only; an arbitrary user stroke is covered by
nothing (hence the "sandbox — qualitative only" badge). A drawn-wedge test
against θ-β-M closes it.

## 3. Case save/load

Cut from the PoC; everything is driven from sliders. First thing to add back.

## 4. Local timestepping

Cut, replaced by the turbo control. Biggest single steady-state win (~30 min
of work on the next pass).

## 5. Grid-convergence study

One resolution shipped; the measure grid exists so the same case can be run
twice and the numbers compared. T8's N_throat 20/40 pair is the start of this.
