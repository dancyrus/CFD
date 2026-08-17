# Work order C1: convergence by trailing-mean plateau

GOAL
Replace the finish-line criterion with a plateau on the trailing MEAN of
a monitored signal, sampled at fixed physical-time intervals, with the
window auto-sized from the domain transit time. Keep the density
residual as a reported diagnostic — it is no longer the finish line and
must not be deleted.

WHY THE RESIDUAL CRITERION HAD TO GO (correctness, not performance)
RESIDUAL_CONVERGED = 1e-3 never fires on the demo case. Measured over
12,000 steps the normalised density residual bottoms at 8.2e-2 around
step 7,750, then RISES to 1.06e-1 and oscillates: a mildly overexpanded
sea-level plume breathes and has no steady state. Since `converged`
gates the report (app.rs greys every integrated quantity; the Report
carries Confidence::NotConverged), the user could never see a valid
report on the default case.

WHY MEAN-PLATEAU, NOT PEAK-TO-PEAK
A peak-to-peak spread over a trailing window was designed first and
measured against the mean-plateau on the same unmodified baseline:
peak-to-peak settles the demo case at step 5,526, the trailing-mean at
step 2,751 — 2.0x on wall clock for zero solver work. Most of those
extra steps were the monitor waiting out the plume's breathing
amplitude; the trailing mean is stationary as soon as the transient
decays. Per-machine re-measurements: docs/results/convergence-<machine>.json,
written by `cargo test -p cfd-core --test convergence -- --include-ignored`.

THE SIGNAL (measured, not assumed)
Three candidates were surveyed on the demo case, per-window means and
drifts (docs/results/convergence-<machine>.json):
- int(rho u_z dV), whole domain: still drifting 3e-2/window at step
  7,300, halving every ~2 windows — it tracks the FULL acoustic
  equilibration of the far field (the ~15,800-step figure in §9), not
  the report. Rejected.
- lip mass flow: shows isolated one-window dips below any tight
  threshold (1.7e-4) while later windows drift 1.7e-2 — a latch on a
  lucky dip, not a plateau. Rejected as the single gate.
- exit-plane THRUST: window-mean drift decays into a stable
  ~1e-3..8e-3 band by ~2.5 transits and stays there (the breathing is
  undamped). Chosen: it is the report's headline number and the most
  plume-sensitive report quantity, so when its mean has stopped moving
  the rest of the report has too. No wall in the domain -> the report
  is empty (NaN) -> the monitor never fires: nothing to report,
  nothing to settle.

THE CRITERION (cfd-core/src/monitor.rs, wired in step.rs)
1. Signal: the report's exit-plane thrust (above).
2. Sample at fixed PHYSICAL TIME intervals, never step counts: dt
   varies during a run, and under local time stepping (deferred, §0)
   step count stops meaning anything. SAMPLES_PER_WINDOW = 16.
3. Window auto-sized from the domain transit time:
   window = 0.5 * L_z / (|u|+a)_ref, with the reference speed the
   isentropic maximum sqrt(gamma (gamma+1)/(gamma-1)) (nondim,
   a0^2 = gamma). No hardcoded step counts anywhere.
4. Fire when the window mean has moved less than tol/2 = 1e-2
   (relative, i.e. under 1% per half-transit) from the previous
   window's mean (CONVERGED_TOL = 2e-2). Sized from measurement: the
   settled drift band is ~1e-3..8e-3 (anything below ~1e-2 fires on a
   lucky beat or never), and the gated number carries +/-5% mass-flow
   quantization and ~13% staircase thrust bias at the default 20
   cells/r_t, so 1%/window is 5-13x tighter than the number it
   protects. A verdict needs two full windows, so the earliest fire is
   one whole domain transit — the startup front cannot outrun it.
5. The verdict LATCHES until set_ambient, set_geometry (at its drain
   point) or a fresh initial field re-arms the monitor, so a borderline
   window cannot flicker the report.
6. The sampling clock gates the work: the report is built ~1 step in
   50, so the monitor costs nothing measurable.

WHAT THIS DID NOT CHANGE
- StepInfo::residual stays: same definition (L2 of the density update,
  normalized by its step-10 value, NaN before step 10), now documented
  as a diagnostic. The UI residual meter stays, relabelled diagnostic.
- StepInfo's shape and the Solver trait are untouched; `converged`
  changed meaning only. Contract-change rule followed: docs/contract.md
  and docs/physics-reference.md §9 updated in the same commit, full
  acceptance ladder re-run (ladder + acceptance, --include-ignored).
- cfd-core/src/physics.rs untouched (work order A2 owns it in
  parallel). The monitor lives in its own new file, monitor.rs.

MEASUREMENT (docs/results/convergence-<machine>.json)
cfd-core/tests/convergence.rs races all three criteria on one demo-case
run: the shipped mean-plateau (from StepInfo.converged), the replaced
peak-to-peak design re-implemented on the same signal/window/clock, and
the residual criterion (recorded minimum; asserted to NEVER fire — if
it ever does, the premise of this work order changed and that is a
report, not a silent pass).
