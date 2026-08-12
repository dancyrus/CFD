# Session E — Integration

**Budget: 25 minutes of merging, then unbounded physics debugging.**

This is the coordinator session, still alive. Do not start a fresh one — you are the only session holding the contract in context and the only one that never needs re-briefing. Run from the **main checkout**, not a worktree.

## Your job is four things and nothing else

1. Merge the branches in dependency order, running the crate's tests after each.
2. Make `cargo test --workspace` green. You may fix call sites. You may not change physics.
3. Swap `MockSolver` for `EulerSolver` in `cfd-ui`.
4. Run the acceptance ladder and report the numbers to the human.

## Explicit prohibitions

- **You may not modify `cfd-contract/`.** If a merge conflict lands there, something went wrong upstream; report it, do not resolve it.
- **You may not fix a failing physics test by loosening a tolerance.** A failing validation case gets reported to the human, not silenced. The tolerances in `docs/physics-reference.md` §12 were set against independently recomputed references; if one fails, the solver is wrong, not the threshold.
- You may not "improve" a working crate. Merge, test, move on.

## Order

Never merge all four at once. Merge in dependency order so any failure is unambiguously attributable.

```bash
git merge core-kernel  && cargo test -p cfd-core -- --ignored
git merge core-physics && cargo test -p cfd-core
git merge geom         && cargo test -p cfd-geom
git merge ui           && cargo run -p cfd-ui
```

`cfd-geom` depends on nothing but the contract, so it cannot break anyone — but the two solver halves are the risk, so they go first and separately. Session A merging alone means a failure there is A's. Session B merging second means the first *real* integration bug shows up in a known place.

Because the coordinator already wrote `cfd-core/src/step.rs` and `lib.rs`, and neither A nor B ever opened them, the two solver merges should be textually conflict-free. If they are not, someone edited a frozen file — find out who before proceeding.

On a `Cargo.lock` conflict, regenerate rather than resolve: `rm Cargo.lock && cargo generate-lockfile`.

## Swapping the mock

One line in `cfd-ui`: `Box::new(EulerSolver::new(setup)?)` instead of `Box::new(MockSolver::new(setup)?)`. Keep the `SolverKind` flag wired so the human can flip back without a rebuild.

## Acceptance ladder

Run in order, cheapest first. Report every number, not just pass/fail — the human is deciding whether to keep going, and "T8 mass flow came out 0.91 of ideal" is actionable where "T8 failed" is not.

| | What | Pass |
|---|---|---|
| T0 | rasterizer area, no flow | ≤ 0.5% at every R |
| T1 | free-stream preservation | ≤ 1e-5 |
| T2 | conservation drift, f64 diagnostics, **measured against the flip ledger** | ≤ 2e-6 |
| T3 | order of accuracy, smooth advection | unlimited ≥ 1.90, limited ≥ 1.50 |
| T4 | Sod, 1-cell strip | L1(ρ) ≤ 6.0e-3 |
| T5 | positivity | floor activations **= 0** |
| T6 | wedge vs θ-β-M | β = 45.34° ± 1.5° |
| T7 | cone vs Taylor–Maccoll | β = 26.74° ± 1.5°, surface p/p∞ = 1.374 ± 8% |
| T8 | nozzle vs isentropic | ṁ/ṁ_ideal ∈ [0.94, 1.00] |

Two things to know while reading results:

**T5 is a hard stop, not a soft failure.** A nonzero floor counter means every downstream number is un-auditable. Do not report T6 through T8 if T5 failed.

**T7 has a trap in the literature.** The NASA NPARC archive lists surface p₃/p₁ = 1.4234 for the M = 2.35, 10° cone. That value is wrong for a 10° cone — it corresponds to a 10.791° cone, and the archive's own β and p are mutually inconsistent. Its *Mach* numbers are correct to five digits. The value in our test, 1.373936, was independently recomputed by integrating Taylor–Maccoll. **Do not "fix" it back to the archive's.**

## Predicted first-integration failures

| Symptom | Look here first |
|---|---|
| Plume points up, or the nozzle is mirrored | someone open-coded an index instead of using `Grid::idx`, or flipped image rows outside `cfd-ui` |
| Thrust off by 1e3 or 1e9 | a length in mm, or a missing non-dimensional conversion at the `Snapshot` boundary |
| Mass flow off by an area factor | someone wrote their own cell volume instead of `Grid::cell_vol` |
| NaN on step 1 | inlet `D` clamp missing, or `mach_from_area_ratio` at `ar = 1` |
| Faint bright or dark line on the axis | the radial flux difference and the axisymmetric source were accumulated separately in f32 — check that `step.rs` still has them in one bracket |
| Standing wave pattern in the plume | the sponge is the per-step form instead of the dt-based one |
| Length mismatch panic | someone crossed a crate boundary with a ghost-padded array |
| Residual plot noisy, colours flicker | a reduction accumulating in f32 |

## Reporting to the human

One table: each test, expected, actual, pass/fail. Then one sentence on what you would look at next. Do not summarize the code. Do not propose refactors.
