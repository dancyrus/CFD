# Work order: general-geometry acceptance rungs (A1)

**Status:** rungs landed; three of the four fail against the solver as it
stands. That is the intended outcome — they were added *before* the fixes, not
after. Nothing in the solver was changed by this work order and no tolerance
was weakened to make a rung pass.

## Why

The acceptance ladder has two non-nozzle rungs, T6 (15° wedge) and T7 (10°
cone). Both bodies are

- **single** — one connected solid component,
- **thick** — tens of cells across in the wall-normal direction,
- **steady** — they converge to a stationary solution,
- **attached to the domain edge** — the wedge grows out of the r = 0 boundary,
  the cone out of the axis.

So they satisfy every assumption a general sandbox breaks. This project is a
sandbox that must eventually simulate arbitrary drawn geometry, which means
optimizations must never be tuned for nozzles — and right now nothing in the
ladder would notice if one were. A per-column bore scan, a "coarsen away
features under N cells" rule, a fast path that assumes one solid span per row,
a convergence classifier that assumes convergence exists, a SIMD re-baseline
that quietly adds dissipation: every one of those passes T0–T8 today.

These four rungs exist to make "not tuned for nozzles" testable rather than
promised. Each breaks exactly one of the four assumptions, and G0 breaks the
one that has no geometry in it at all.

| Rung | Breaks | Exact reference exists because |
|---|---|---|
| G1 | single body | two-shock theory is exact for a symmetric double wedge |
| G2 | thick body | shock-expansion theory is exact for a diamond at zero incidence |
| G3 | steady, and `report()`'s whole vocabulary | the published solution is structural, so the assertions are structural |
| G0 | needs no geometry assumption at all | d'Alembert: the answer is exactly zero |

All four reuse the T6 rig — uniform planar grid, free stream at `M_inf`,
stagnation chamber set to the stream's own stagnation state — and differ only
in the solid mask. They live in `cfd-core/tests/ladder.rs` alongside T2–T8 and
run with the same command.

### What was deliberately NOT added

Cylinder drag coefficient, Strouhal number, backward-facing step reattachment
length, pipe friction factor. This is an inviscid Euler solver with no Reynolds
number. Cylinder `C_d` and reattachment length are viscous separation
phenomena and the exact *inviscid* answer for cylinder drag is zero; a Strouhal
number here would be set by numerical dissipation and would be mesh-dependent;
friction factor is wall shear, and §13 already refuses to display shear. All
four would be meaningless assertions that look authoritative.

G0 uses an ellipse and asserts the zero and its convergence rate. It is not a
cylinder-drag test and must never be turned into one.

## New solver code: `cfd-core::forces`

G2 asserts a drag coefficient, which needs a surface pressure integral over the
fluid/solid faces of **one** body. The solver did not have one. Work orders A3
and C2 need the same integral, so it was built to be reused rather than folded
into the test:

- `forces::label_bodies(&SolidField) -> Bodies` — connected components,
  4-connected on interior indices (the same neighbourhood the fluxes use, so
  corner contact is two bodies), plus `Bodies::selector(id)`.
- `forces::surface_pressure_force(w, solid, gas, geometry, body) -> SurfaceForce`
  — `f_z`, `f_r`, wetted area, face count.
- `EulerSolver::primitives()` — interior primitives in the solver's own
  non-dimensional units, which is what the integral wants (`Snapshot` is the SI
  display copy and a lossy place to do physics from).

The integrand is `physics::wall_flux_{z,r}` itself, area-weighted — the same
oriented wall momentum flux the sweeps subtract from the adjacent fluid cell,
not a fresh `p·n` quadrature. That equality is the point: it makes the force
and the momentum the fluid actually loses the same quantity, which is the only
reason a momentum balance over a domain containing a body can close at all.
G3 closes exactly that balance, so the definition is guarded, not asserted.

Definition and sign convention: `docs/physics-reference.md` §11. Signatures:
`docs/contract.md`. This is a contract change under the CLAUDE.md rule — both
documents are mirrored in the same commit and the full ladder was re-run.

## The rungs

### G1 — multi-body regular reflection

Symmetric double wedge: two 10° wedges facing each other across a 2.0-wide
channel, tips at z = 0.3, `M_inf = 2.0`, γ = 1.4, planar. **The solid mask has
two disconnected components, which is the point.** Any "one connected solid" or
per-row-span assumption reads this geometry as one body, or merges the two
spans in a column, or scans to the first solid cell and stops.

Exact two-shock reference, recomputed here from θ–β–M and Rankine–Hugoniot
(bisection on the weak branch, `docs/work-orders/general-geometry-refs.py` reproduces every digit; nothing is
taken from a chart):

| | value |
|---|---|
| incident β₁ | 39.313932° |
| p₂/p₁ | 1.706579 |
| M₂ | 1.640522 |
| reflected shock, to the LOCAL flow | 49.384042° (39.384042° to the axis) |
| p₃/p₂ | 1.642579 |
| **p₃/p₁** | **2.803191** |
| M₃ | 1.284889 |

Flow downstream of the reflection is parallel to the axis again.

Asserted: the component count is 2; p₃/p₁ sampled in region 3 near the
midplane; and the reflection point, obtained by fitting the incident shock over
rows r ∈ [0.60, 0.90] (clear of both staircase tips and of the reflection
interaction) and intersecting the fit with the midplane. Exact reflection
point z = 0.3 + 1.0/tan β₁ = 1.521157.

### G2 — thin body

Symmetric diamond airfoil, chord 1, 5° half-angle, zero incidence,
`M_inf = 2.0`, γ = 1.4, immersed mid-channel (it touches no boundary). At
h = 0.01 — 100 cells of chord — the body is 8.7 cells thick at mid-chord, so it
catches coarsening rules that delete thin features, and it degrades the MUSCL
stencil on *both* walls of the same fluid column.

The duct walls do not interfere: the leading-edge shock reflects and re-crosses
the chord line at z = 2.36, well past the trailing edge at 1.6, and the flow is
supersonic so nothing propagates upstream. The surface pressures are free-air
values.

Shock-expansion reference (recomputed by the same script):

| | value |
|---|---|
| t/c | 0.087489 |
| leading-edge shock β | 34.301575° |
| fore-surface p/p∞ | 1.315407 |
| aft-surface p/p∞ | 0.747760 |
| **wave drag C_d** | **0.017737** |
| Ackeret linearised | 0.017677 (0.34% low — that is the linearisation error) |

Asserted: `C_d`, through `forces::surface_pressure_force` over the airfoil
component alone (the top wall is the other component and is excluded by
selector, not by index).

### G3 — never steady

Woodward & Colella's Mach 3 wind tunnel with a forward-facing step: a 3 × 1
duct, step 0.2 tall starting at z = 0.6, uniform M = 3 inflow, γ = 1.4.
Reference: <https://www.cfd-online.com/Wiki/2-D_Mach_3_Wind_Tunnel_With_a_Step>.

`report()`'s lip and throat logic is meaningless here by construction — there
is no throat, the "lip" it detects is the downstream end of the step, and every
number it derives from them is nonsense. That is what makes the case valuable:
it is the only rung where the engineering report has nothing to say and the
solver still has to be right.

Both duct walls are **solid rows** rather than the mirror boundary, so every
wall force appears in the surface integral and the momentum balance closes.

The published solution is structural, so the assertions are structural:

1. **Positivity floor activations stay at zero.** The convex corner at the top
   of the step is a singular expansion and is where a two-level positivity
   fallback fails if it is wrong.
2. **Mass, z-momentum and energy balance against the analytic inflow** to T2's
   2e-6, over the window before any disturbance reaches the outflow. Inside
   that window both boundary fluxes are exactly the free-stream flux, so the
   balance carries *no discretization error at all* — only f32 roundoff against
   f64 reductions. The window is verified, not assumed: the test asserts the
   exit column is still free stream before it believes the numbers. The
   momentum balance additionally needs the wall impulse, accumulated
   trapezoidally from `surface_pressure_force`.
3. **The Mach stem forms on the leading bow shock**: at the top wall the
   leading front is normal (its z position is constant across the six rows
   below the wall) and the flow behind it is subsonic. Both together are what
   "Mach stem" means; either alone is weaker than it looks.

It deliberately does **not** assert convergence. A later work order adds the
assertion that the convergence classifier never returns STEADY on this case.

### G0 — negative control

`M_inf = 0.3` over a smooth body: half of a 2:1 ellipse on the r = 0 symmetry
plane inside a duct with a solid top wall, so the mirrored picture is a smooth
closed body centred in a 2.5-tall channel. Steady subsonic inviscid flow past a
fore-aft symmetric body in a symmetric duct is fore-aft symmetric, so **the
exact drag is exactly zero** (d'Alembert). Nothing about the answer is
approximate: every count the solver reports is its own dissipation, and it must
fall roughly linearly with cell size.

Asserted: `C_d` decreases monotonically over h = 0.05 / 0.025 / 0.0125, and the
observed order on the fine pair is ≥ 1.0. Plus floors = 0 and max Mach < 0.95,
so "shock-free" is checked rather than assumed.

This is the cheapest test that catches an optimization quietly adding
dissipation, and it is what will guard a later SIMD re-baseline: a re-baseline
that changes the dissipation shows up here as a changed `C_d` at fixed h even
when every shock-capturing rung still passes.

## Where the tolerances come from

Derived from grid-convergence studies at three cell sizes, the way T4's Sod
threshold and T6's ±1.5° were derived — never guessed, never fitted to what the
solver happens to produce.

Derived tolerances, the way T4's Sod threshold was set in the measured gap
between second- and first-order error rather than chosen. Neither is fitted to
what the solver happens to produce, and neither may be widened.

**G1 — the ±1.5° shock-angle claim, propagated.** This document set (§13, and
T6's and T7's own bands) claims ±1.5° on any captured shock angle. Pushing
±1.5° on the incident β₁ through the exact two-shock solution gives:

| β₁ | θ implied | p₃/p₁ | rel err | reflection point | err |
|---|---|---|---|---|---|
| 37.8139° | 8.5881° | 2.43600 | 13.10% | 1.58855 | 0.0674 |
| **39.3139°** | **10°** | **2.803191** | — | **1.521155** | — |
| 40.8139° | 11.3436° | 3.22902 | 15.19% | 1.45794 | 0.0632 |

The rung takes the tighter side of each — **13.1%** on p₃/p₁ and **0.063** on
the reflection point — so it is never laxer than the claim it inherits.

**G2 — the staircase's own floor.** An immersed staircase (physics-reference
§3) quantizes a body's *projected frontal area* to one cell. A diamond of
thickness t = c·tan 5° therefore carries a frontal-area error up to 2h/t no
matter how good the flow solution is: 45.7% at h = 0.02, **22.9% at h = 0.01**,
11.4% at h = 0.005. That is the tightest C_d tolerance this architecture can be
held to, so it is what the rung asserts. Anything missed beyond it is the
scheme, not the geometry — which is precisely what the study below shows.

**G3 and G0** assert structure and a convergence rate, so neither needs a
value tolerance. G3's 2e-6 is T2's, unchanged: inside the pre-disturbance
window the balance carries no discretization error at all, so there is nothing
to loosen it for.

### Grid-convergence studies

Run at three cell sizes each before the tolerances were written down.

**G1** (fit window r ∈ [0.60, 0.90] fixed in physical space; the reflection
point is measured directly at the midplane row, not extrapolated from the fit):

| h | p₃/p₁ | rel err | reflection-point err | fitted β₁ | β₁ err |
|---|---|---|---|---|---|
| 0.02 | 4.4400 | 58.4% | −0.3691 (−18.5 cells) | 45.098° | +5.784° |
| 0.01 | 3.8410 | 37.0% | −0.2368 (−23.7 cells) | 41.726° | +2.412° |
| 0.005 | 2.9843 | 6.5% | −0.1396 (−27.9 cells) | 40.258° | +0.944° |

Observed order: 1.26 / 1.35 on the shock angle, 0.64 / 0.76 on the reflection
point. Floors are zero and the component count is 2 at every level.

**G2**:

| h | cells of chord | body thickness | C_d | rel err | bound 2h/t | fore p/p∞ | aft p/p∞ | wetted faces |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 50 | 4.4 cells | 0.10090 | 469% | 45.7% | 1.4597 | 1.2786 | 84 |
| 0.01 | 100 | 8.7 cells | 0.07827 | 341% | 22.9% | 2.3184 | 1.0438 | 192 |
| 0.005 | 200 | 17.5 cells | 0.05481 | 209% | 11.4% | 1.7400 | 0.8664 | 412 |

Observed order of the C_d error: 0.46 / 0.71 — *slower* than the first-order
quantization bound, so the gap against what the geometry allows widens under
refinement: 10.3× the bound at h = 0.02, 14.9× at 0.01, 18.3× at 0.005. The
surface pressures are recorded, not asserted; they are sampled in the first
fluid cell above one column and a staircase makes that sample step-phase
dependent, which is why the h = 0.01 fore value (2.32 against an exact 1.32)
is not monotone with the others.

**G3** — a dt refinement, to establish which side of the balance the momentum
residual lives on:

| h | CFL | mass | z-momentum | energy | wall impulse |
|---|---|---|---|---|---|
| 1/80 | 0.4 | 4.298e-7 | 5.388e-5 | 1.231e-7 | 1.05663 |
| 1/80 | **0.2** | 1.198e-6 | **1.179e-5** | 9.776e-8 | 1.05538 |
| 1/160 | 0.4 | 3.656e-7 | 2.699e-5 | 7.670e-8 | 1.05661 |

Halving CFL at fixed h halves dt and divides the momentum residual by 4.57 —
second order in dt, which identifies it as the trapezoid quadrature of the wall
impulse rather than a solver imbalance. See the findings below. Halving the
mesh at fixed CFL also halves dt but changes the force history at the same
time, so it is the CFL row that is diagnostic. Note the mass residual *rises*
when CFL halves: twice the steps, twice the f32 roundoff, still inside 2e-6.


## Results

### Which rungs fail, and why

**G1 — FAILS, both assertions.** At the ladder's h = 0.01: p₃/p₁ = 3.841
against an exact 2.803 (37.0% high, tolerance 13.1%), and the reflection point
sits 0.237 upstream of exact (23.7 cells, tolerance 0.063). Two disconnected
components are handled correctly — the mask is labelled as 2 bodies, both
wedges generate their shock, and the mirror-row symmetry check passes — so
nothing here is a multi-body bug. **The cause is the staircase wall
over-turning the flow at a shallow wall angle.** The fitted incident β is
41.73° against an exact 39.31°; inverting θ–β–M on the measured angle says the
solver is turning the stream 12.1°, not 10°, and feeding that θ through the
exact two-shock solution reproduces the measured p₃/p₁ to within a few percent.
The error compounds because it is applied twice, once per shock.

The bias is a function of wall angle, not just of h: T6's 15° wedge measures
β 1.10° high at the same cell size where this 10° wedge measures 2.41° high.
The shallower the wall, the further apart the staircase steps and the more each
one behaves like an isolated blunt obstacle. Nothing in the ladder measured
that before, because T6 and T7 are the only wall-angle rungs and both are
steeper.

Halving the mesh to h = 0.005 brings p₃/p₁ inside tolerance (6.5%) but leaves
the reflection point at 0.140, still 2.2× over. At the observed 0.7 order that
would need h ≈ 0.0015 — a 1700 × 1300 grid. **The reflection-point assertion is
not reachable by refinement at any affordable cost; it needs the wall
treatment fixed.**

**G2 — FAILS.** C_d = 0.0783 against an exact 0.0177 at h = 0.01: 341% high,
against a 22.9% tolerance. Drag is over-predicted by a factor of 4.4, and the
refinement study shows the error decaying at order 0.46–0.71, *slower* than the
first-order frontal-area quantization bound, so the gap against what the
geometry allows gets relatively worse under refinement (10.3× the bound at
h = 0.02, 18.3× at h = 0.005). The mechanism is the staircase again but in its
other mode: each step on the fore surface is a small forward-facing face where
the flow locally stagnates, so the summed pressure on the z-normal faces is far
above the true surface pressure, while the rearward-facing steps on the aft
surface sit in a base-pressure-like state. The surface pressure integral itself
is sound — G3's momentum balance closes against it to 4e-7 on mass and 1e-7 on
energy, which is the strongest available check that the force and the momentum
the fluid loses are the same quantity.

**G3 — FAILS on one leg of three.** Mass balances to 4.30e-7 and energy to
1.23e-7 against T2's 2e-6, floor activations are zero through the whole run,
and the Mach stem is unambiguous: the leading front on the top wall spreads
0.32 cells across the six near-wall rows (an oblique reflection would spread
6), and the flow behind it drops to Mach 0.454. The z-momentum balance comes in
at 5.39e-5, 27× over.

**That residual is the measurement's, not the solver's.** SSP-RK2 applies
0.5·dt·(F_wall(u₀) + F_wall(u₁)) of wall impulse per step, and u₁ — the stage-1
state — is not observable from outside the solver, so the test trapezoids the
impulse across steps instead. That approximation is second order in dt, and the
CFL refinement above confirms it exactly: halving dt at fixed mesh divides the
residual by 4.57. Reaching 2e-6 by this route would need CFL ≈ 0.08. The
assertion is left at T2's 2e-6 rather than widened to something the quadrature
can hit, because widening it would be exactly the "loosen a tolerance until it
passes" move this repo forbids. **The fix is in the solver, not in the test:
accumulate the wall impulse inside `step()`, where both stage forces exist, and
the balance closes to the same roundoff floor as mass and energy.** That is a
change to `step.rs` and is out of scope for this work order, which adds rungs
and changes no solver behaviour.

**G0 — FAILS, and the failure is the most interesting one.** It does not fail
on the drag decay. It fails on the *premise*:

| h | C_d (mean of last 10 of 40 crossings) | peak-to-peak C_d | settled | residual |
|---|---|---|---|---|
| 0.1 | +7.13e-1 | 1.20e-2 | **yes** | 1.59e-4 |
| 0.05 | +6.45e-1 | 6.07e-1 | no | 5.75e-3 |
| 0.025 | +5.25e-1 | 1.01e0 | no | 5.48e-3 |

At the coarsest mesh the flow settles by this project's own definition
(normalized residual < 1e-3, physics-reference §9) and reports a steady
C_d = 0.71 where the exact answer is 0. **As the mesh refines the flow stops
settling.** The peak-to-peak swing in C_d over the averaging window grows from
1.2e-2 to 1.0 — larger than the mean itself — and running longer makes it
worse, not better: at h = 0.05 the residual goes 9.6e-3 → 5.7e-3 → 2.3e-2 at 20
→ 40 → 80 crossings, and the mean C_d swings +0.65, −0.52, −0.0004. This is a
solver with no viscosity, no Reynolds number and no shock anywhere in the field
(max Mach 0.55) producing an oscillating separated wake behind a streamlined
ellipse. Inviscid flow has no mechanism for it; §13 already states the
simulation "cannot separate."

So the drag number is not a measurement at h ≤ 0.05, and the convergence rate
cannot be extracted from an oscillation. The rung reports it that way: it
asserts steadiness first, with a message that says so, and only then the decay
rate. Both the magnitude (C_d = 0.71 at the one steady level, against an exact
0) and the onset of unsteadiness under refinement are recorded.

### Summary

| Rung | Asserted | Result |
|---|---|---|
| G1 | 2 components | **pass** |
| G1 | p₃/p₁ within 13.1% | **fail** — 37.0% at h = 0.01 |
| G1 | reflection point within 0.063 | **fail** — 0.237 at h = 0.01 |
| G2 | C_d within 2h/t = 22.9% | **fail** — 341% at h = 0.01 |
| G3 | floors = 0 | **pass** |
| G3 | mass balance ≤ 2e-6 | **pass** — 4.30e-7 |
| G3 | energy balance ≤ 2e-6 | **pass** — 1.23e-7 |
| G3 | z-momentum balance ≤ 2e-6 | **fail** — 5.39e-5, quadrature-dominated |
| G3 | Mach stem normal at the top wall | **pass** — 0.32 cells of spread |
| G3 | subsonic pocket behind the stem | **pass** — min Mach 0.454 |
| G0 | floors = 0, shock-free | **pass** |
| G0 | flow is steady, so d'Alembert applies | **fail** — settles only at h = 0.1 |
| G0 | C_d falls at first order or better | **not reached** — the premise fails first |

Nothing was fixed in this session and no tolerance was weakened. T0–T8 were
re-run unchanged as the contract-change rule requires; their results are in
`docs/results/`.

### What the failures point at

All four failures have one root: **the immersed staircase wall**
(physics-reference §3). G1 is its over-turning bias at shallow wall angles, G2
its stagnation pressure on step faces, G0 its numerically-separated wake. Only
G3's is unrelated, and G3's is in the test harness rather than the solver. That
is a useful result on its own — it says the ladder's four new failure modes are
one problem, not four, and that the wall treatment is where optimization effort
belongs before any of the shortcuts these rungs were written to guard against.


## Reproducing

```
cargo test -p cfd-core --test ladder -- --include-ignored --nocapture
cargo test -p cfd-core --test acceptance -- --include-ignored --nocapture
```

Measured numbers land in `docs/results/ladder-<machine>.json` automatically,
from inside each rung *before* its own asserts, so a failing rung still records
its value with `"pass": false`.
