# CFD Sandbox — proof of concept

A 2D axisymmetric compressible CFD sandbox built on real numerical methods. Draw a
shape or dial in a rocket nozzle, watch supersonic flow run through it, drag an
altitude slider and watch the plume change.

This repository started as the **plan**; the code now exists, written by parallel
Claude Code sessions working from the briefs in `docs/sessions/`.

## Download

Prebuilt binaries for Windows (x86_64) and Apple Silicon macOS are on the
[releases page](https://github.com/dancyrus/CFD/releases): unzip and run — the
app is a single executable, no installer.

- **Windows**: run `cfd-ui.exe`.
- **macOS (Apple Silicon)**: the binary is ad-hoc signed but not notarized,
  so the first launch needs `chmod +x cfd-ui`, then right-click → Open (or
  `xattr -d com.apple.quarantine cfd-ui`).

Everything else (Linux, Intel macOS) builds from source:
`cargo run --release -p cfd-ui`.

## What it does

Finite volume, MUSCL reconstruction with a slope limiter, HLLC Riemann solver,
SSP-RK2, on one uniform anisotropic Cartesian axisymmetric grid. Geometry — both a
parametric nozzle contour and a hand-drawn wall — rasterizes into a single solid
volume-fraction field, so one solver serves both. CPU with rayon; egui for the
interface.

**It will:** get area-ratio physics, exit Mach and thrust coefficient right; capture
sharp shocks and expansion fans; show the 2D divergence loss quasi-1D misses; produce
an under-expanded plume with shock-cell structure and a Mach disk; run drawn geometry
and a parametric nozzle through the same solver.

**It will not:** resolve a boundary layer, compute wall heat flux, model turbulence,
or separate. Mass flow carries about ±5% from the staircase wall at the shipping
grid. There is no entrainment, so the plume edge stays razor-sharp and shock cells
persist further downstream than in a photograph.

Those limits are enforced in the interface, not just documented — see
`docs/physics-reference.md` §13.

## Layout

```
CLAUDE.md                      agent instructions, read first
docs/
  physics-reference.md         every formula, constant, tolerance. The authority.
  contract.md                  exact type and function signatures between crates
  build-plan.md                the human's runbook: what to type, when, what to check
  sessions/
    00-coordinator.md          blocking phase: scaffold, kernels, orchestration, mock
    A-core-kernel.md           sweeps, timestep, rayon, positivity
    B-core-physics.md          axisymmetric source, wall, BCs, sponge, initialization
    C-geom.md                  contours, rasterizer, editor data model
    D-ui.md                    the app
    E-integration.md           merge order, acceptance ladder, predicted failures
```

Crates, once the coordinator has run: `cfd-contract` (frozen types and pure kernels),
`cfd-core` (solver), `cfd-geom` (geometry), `cfd-ui` (app).

## How to build it

Read `docs/build-plan.md`. In outline: one 50-minute coordinator session writes the
frozen contract, HLLC and MUSCL, the RK2 orchestration, the acceptance tests and a
mock solver. Then four sessions run in parallel in git worktrees, each owning one
file or crate and none able to edit another's. Then the coordinator merges.

Every physics number in these documents was independently recomputed by a separate
verification pass — exact Riemann solver, Taylor–Maccoll integration, symbolic
identity checks, 18,000-sample Newton sweeps. Where a published reference turned out
to be wrong, the document says so and gives the corrected value.

## Schedule, honestly

| | P50 |
|---|---|
| UI running, geometry editor working, mock plume animating | 2h 15m |
| Drawn geometry with real compressible flow moving through it | 3h 10m |
| A nozzle plume a propulsion engineer would not laugh at | 3h 50m (80% CI 3h05m – 6h30m) |

The plan is built so that anything after minute 120 is demoable, and there is a
four-rung abort ladder in `docs/build-plan.md` §5. Every rung is a flag written at
minute 25, not code written at minute 150.
