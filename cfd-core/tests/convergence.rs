//! Work order C1 — before/after measurement for the convergence criterion
//! swap. Results get committed, not reported in chat (CLAUDE.md): every
//! number below is recorded to `docs/results/convergence-<machine>.json`
//! BEFORE the asserts.
//!
//! Three criteria are raced on one demo-case run (the app's default: gamma
//! 1.24, area ratio 8, sea level, 320x200 compact domain):
//!
//! - **after** — the shipped criterion: plateau on the trailing MEAN of the
//!   report's exit-plane thrust, window auto-sized from the domain transit
//!   time, sampled on physical time (`cfd_core::monitor`, wired into
//!   `EulerSolver::step`).
//! - **before** — the design it replaced: peak-to-peak spread of the same
//!   signal over the same trailing window, fire when the spread drops under
//!   tol relative. Re-implemented here, on the same sampling clock, so the
//!   two step counts are directly comparable.
//! - **residual** — the criterion before either: normalised density residual
//!   < 1e-3. Documented correctness bug: the sea-level plume breathes, the
//!   residual bottoms out around 1e-1 and oscillates, so this NEVER fires and
//!   `Confidence::NotConverged` permanently blanked the demo-case report.
//!
//!     cargo test -p cfd-core --test convergence -- --include-ignored --nocapture

use std::sync::Arc;

use cfd_contract::{
    Ambient, Chamber, GasModel, Grid, Numerics, RefScales, SolidField, SolveSetup, Solver,
};
use cfd_core::monitor::{window_time_for, CONVERGED_TOL, SAMPLES_PER_WINDOW};
use cfd_core::EulerSolver;
use cfd_results::{record_note, record_test, TestResult, Value};

const SUITE: &str = "convergence";
const MAX_STEPS: u64 = 8000;
const TOL: f64 = CONVERGED_TOL; // both criteria get the same budget

fn record(id: &str, name: &str, expected: impl Into<Value>, actual: impl Into<Value>,
          units: &str, pass: bool) {
    record_test(SUITE, TestResult {
        id: id.into(), name: name.into(), expected: expected.into(),
        actual: actual.into(), units: units.into(), pass,
    });
}

/// The demo case exactly as the app ships it: 320x200 uniform compact domain
/// (46.4 x 10 r_t), 15° cone at area ratio 8, gamma 1.24, sea level, default
/// numerics (MUSCL/HLLC, sponge 24, quasi-1D init). Same setup as
/// diag_nozzle_centerline.
fn demo_setup() -> SolveSetup {
    let gas = GasModel { gamma: 1.24, r_specific_si: 378.0 };
    let refs = RefScales::from_chamber(0.05, 5.0e6, 3200.0, &gas);
    let grid = Grid::uniform(320, 200, 0.1449, 0.05);
    let spec = cfd_geom::NozzleSpec {
        throat_radius_m: 0.05,
        area_ratio: 8.0,
        contraction_ratio: 4.0,
        converge_half_angle_deg: 30.0,
        throat_arc_up: 1.5,
        throat_arc_down: 0.382,
        contour: cfd_geom::ContourKind::Conical { half_angle_deg: 15.0 },
    };
    let wall = cfd_geom::generate_contour(&spec, 512).unwrap();
    let solid: SolidField = cfd_geom::rasterize(&wall, &grid, &refs).unwrap();
    SolveSetup {
        grid,
        solid: Arc::new(solid),
        gas,
        chamber: Chamber { p0: 1.0, t0: 1.0 },
        ambient: Ambient { p: (101_325.0 / refs.p_pa) as f32, t: (288.15 / refs.t_k) as f32 },
        numerics: Numerics::default(),
        refs,
    }
}

/// The replaced design: peak-to-peak spread of the trailing window's samples,
/// relative to their largest magnitude, under the SAME threshold budget the
/// shipped criterion gets — theta = tol/2 ("the signal moved less than theta
/// over the window"). Spread bounds mean-drift from above, so this is the
/// matched comparison: the gap between the two fire steps is exactly the
/// breathing amplitude the trailing mean averages away and the spread has to
/// wait out.
struct PtpMonitor {
    ring: Vec<f64>,
    fired: bool,
}
impl PtpMonitor {
    fn new() -> Self { PtpMonitor { ring: Vec::new(), fired: false } }
    fn update(&mut self, x: f64) -> bool {
        self.ring.push(x);
        if self.ring.len() > SAMPLES_PER_WINDOW { self.ring.remove(0); }
        if !self.fired && self.ring.len() == SAMPLES_PER_WINDOW {
            let lo = self.ring.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = self.ring.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let scale = lo.abs().max(hi.abs());
            self.fired = scale > 0.0 && (hi - lo) <= 0.5 * TOL * scale;
        }
        self.fired
    }
}

#[test]
#[ignore = "measurement: a few minutes of solver time; run explicitly"]
fn c1_before_after_step_counts() {
    let setup = demo_setup();
    let g = setup.grid.clone();
    let window = window_time_for(g.lz(), setup.gas.gamma as f64);
    let sample_dt = window / SAMPLES_PER_WINDOW as f64;
    println!(
        "window {window:.3} (0.5 transit of lz {:.1} at gamma {}), sample every {sample_dt:.3}",
        g.lz(), setup.gas.gamma
    );

    let mut s = EulerSolver::new(setup).unwrap();
    let mut ptp = PtpMonitor::new();
    let mut next_sample = sample_dt;

    let mut after: Option<(u64, f64)> = None; // (step, time) trailing-mean plateau fires
    let mut before: Option<(u64, f64)> = None; // peak-to-peak baseline fires
    let mut res_min = f64::INFINITY;
    let mut res_min_step = 0u64;
    let mut res_fired = false;
    let mut info = s.step().unwrap();
    loop {
        // The shipped criterion, straight off StepInfo.
        if after.is_none() && info.converged {
            after = Some((info.step, info.time));
        }
        // The baseline, on the same physical-time sampling clock and the
        // same signal the shipped monitor watches.
        if info.time >= next_sample {
            next_sample = (next_sample + sample_dt).max(info.time);
            if before.is_none() && ptp.update(s.report().thrust_n) {
                before = Some((info.step, info.time));
            }
        }
        // The residual diagnostic: bottoms out and oscillates, never converges.
        if info.residual.is_finite() && info.residual < res_min {
            res_min = info.residual;
            res_min_step = info.step;
        }
        res_fired |= info.residual.is_finite() && info.residual < 1e-3;
        if info.step >= MAX_STEPS || (after.is_some() && before.is_some()) {
            break;
        }
        info = s.step().unwrap();
    }

    let (a_step, a_time) = after.unwrap_or((u64::MAX, f64::NAN));
    println!("after  (trailing-mean plateau): step {a_step} (t {a_time:.2})");
    match before {
        Some((b, t)) => println!("before (peak-to-peak baseline): step {b} (t {t:.2})"),
        None => println!("before (peak-to-peak baseline): did not fire by step {MAX_STEPS}"),
    }
    println!("residual: min {res_min:.3e} at step {res_min_step}, \
              final {:.3e} at step {} — fired below 1e-3: {res_fired}",
             info.residual, info.step);

    // ---- recorded before any assert ---------------------------------------
    record(
        "C1-mean-plateau",
        "demo case, steps until converged: trailing-mean plateau (shipped criterion)",
        format!("fires within {MAX_STEPS} steps").as_str(),
        if after.is_some() { Value::Num(a_step as f64) }
        else { Value::Str(format!(">{MAX_STEPS}")) },
        "steps",
        after.is_some(),
    );
    record(
        "C1-ptp-baseline",
        "demo case, steps until fired: peak-to-peak spread over the same trailing window, \
         signal and threshold budget tol/2 (replaced design, re-measured on the same run)",
        "recorded — the before number",
        match before { Some((b, _)) => Value::Num(b as f64),
                       None => Value::Str(format!(">{MAX_STEPS}")) },
        "steps",
        true,
    );
    record(
        "C1-residual-criterion",
        "demo case, minimum normalised density residual over the run (old criterion \
         residual < 1e-3 must NOT fire: the plume breathes, there is no steady state)",
        "> 1e-3 at every step",
        res_min,
        "L2(d rho) / step-10 value",
        !res_fired,
    );
    let speedup = match before {
        Some((b, _)) if after.is_some() => format!("{:.2}x", b as f64 / a_step as f64),
        Some(_) => "n/a".into(),
        None => format!(">{:.2}x", MAX_STEPS as f64 / a_step as f64),
    };
    record_note(SUITE, "c1-summary", &format!(
        "Work order C1: convergence criterion swapped from peak-to-peak spread to a plateau \
         on the trailing window MEAN of the report's exit-plane thrust, sampled at fixed \
         physical-time intervals (window = 0.5 domain transit L/(|u|+a)_ref = {window:.3} \
         nondim, {SAMPLES_PER_WINDOW} samples/window, fire when the window mean moves \
         < tol/2 = {:.0e} relative from the previous window's mean). Demo case, this \
         machine: trailing-mean fires at step {a_step} (t {a_time:.2}); the peak-to-peak \
         design on the same signal, window, clock and tol/2 threshold budget fires at \
         {} — {speedup} on steps for zero solver work. The residual criterion (< 1e-3) \
         never fires: minimum over this run {res_min:.2e} at step {res_min_step} of {} — \
         still hundreds of times the threshold when both other criteria had already \
         fired — which is why it gated the report into a permanent \
         Confidence::NotConverged on the default case and had to go. The residual \
         remains a reported diagnostic in StepInfo.",
        0.5 * TOL,
        match before { Some((b, _)) => format!("step {b}"),
                       None => format!("no fire by step {MAX_STEPS}") },
        info.step,
    ));

    // ---- the verdicts -----------------------------------------------------
    let (a_step, _) = after.expect("trailing-mean plateau never fired on the demo case");
    assert!(
        a_step > 1000,
        "plateau fired at step {a_step} — before one domain transit, which means the \
         monitor latched on the startup front, not on convergence"
    );
    assert!(
        !res_fired,
        "the residual criterion fired (min {res_min:.3e}) — the premise of work order C1 \
         (no steady state on the demo case) no longer holds; report it, do not ship blind"
    );
    if let Some((b_step, _)) = before {
        assert!(
            b_step > a_step,
            "peak-to-peak fired at {b_step}, not after the trailing-mean at {a_step} — \
             the swap bought nothing on this machine; report it"
        );
    }
}

/// The signal survey behind the monitor's signal choice (work order
/// C1). Runs the demo case once and prints/records the per-window trailing
/// means and window-to-window drifts of three candidate signals: the
/// domain-integrated axial momentum, the report's thrust, and the report's
/// mass flow. This is the measurement that rejected the momentum integral
/// (tracks full acoustic equilibration, not the report) and the lip mass
/// flow (isolated lucky-dip windows), and that sized CONVERGED_TOL from the
/// thrust drift's settled band.
#[test]
#[ignore = "measurement: a few minutes of solver time; run explicitly"]
fn c1_signal_survey() {
    use cfd_contract::FieldKind;
    use cfd_core::monitor::relative_drift;

    let setup = demo_setup();
    let g = setup.grid.clone();
    let window = window_time_for(g.lz(), setup.gas.gamma as f64);
    let sample_dt = window / SAMPLES_PER_WINDOW as f64;

    let mut s = EulerSolver::new(setup).unwrap();
    let mut next = sample_dt;
    let mut acc = [0.0f64; 3]; // momentum, thrust, mdot
    let mut n = 0usize;
    let mut prev: Option<[f64; 3]> = None;
    let mut rows = Vec::new();
    let mut info = s.step().unwrap();
    while info.step < MAX_STEPS {
        info = s.step().unwrap();
        if info.time >= next {
            next = (next + sample_dt).max(info.time);
            let snap = s.snapshot();
            let mut mom = 0.0f64;
            for ir in 0..g.nr {
                for iz in 0..g.nz {
                    mom += snap.sample(FieldKind::Density, iz, ir) as f64
                        * snap.sample(FieldKind::VelocityZ, iz, ir) as f64
                        * g.cell_vol(iz, ir);
                }
            }
            let rep = s.report();
            acc[0] += mom;
            acc[1] += rep.thrust_n;
            acc[2] += rep.mass_flow_kg_s;
            n += 1;
            if n == SAMPLES_PER_WINDOW {
                let mean = [acc[0] / n as f64, acc[1] / n as f64, acc[2] / n as f64];
                acc = [0.0; 3];
                n = 0;
                if let Some(p) = prev {
                    rows.push(format!(
                        "step {} t {:.1}: mom {:.2e} thrust {:.2e} mdot {:.2e}",
                        info.step, info.time,
                        relative_drift(p[0], mean[0]),
                        relative_drift(p[1], mean[1]),
                        relative_drift(p[2], mean[2]),
                    ));
                    println!("  {}", rows.last().unwrap());
                }
                prev = Some(mean);
            }
        }
    }
    assert!(rows.len() >= 4, "too few windows to survey: {}", rows.len());
    cfd_results::record_note(SUITE, "c1-signal-survey", &format!(
        "Signal survey for the plateau monitor, demo case, window {window:.3} \
         ({SAMPLES_PER_WINDOW} samples): window-to-window relative drift of the trailing \
         mean per candidate signal — [{}]. Read: the momentum integral is still drifting \
         at the end (it tracks full acoustic equilibration of the far field, not the \
         report); lip mass flow shows isolated dips below thresholds its later windows \
         re-cross (lucky-dip risk); thrust decays into a stable breathing band, which is \
         what CONVERGED_TOL/2 sits just above.",
        rows.join("; ")));
}
