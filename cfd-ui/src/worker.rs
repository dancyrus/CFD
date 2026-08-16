//! The solver thread. State flows out through a `triple_buffer` (latest value
//! wins, never blocks either side); commands flow in through `std::sync::mpsc`.
//! Publishes are throttled to at least 16 ms and each one is followed by
//! exactly one `ctx.request_repaint()` — eframe is reactive and will not
//! repaint on its own. See docs/sessions/D-ui.md §1.
//!
//! Step pacing is decoupled from publish cadence: steps track wall clock at
//! `BASE_STEP_RATE * turbo` steps/s (the brief's one-batch-per-16ms-frame
//! semantics), gated by a frame-time budget, so simulated time flows at the
//! same speed regardless of how long `snapshot()` takes. The publish interval
//! additionally adapts to the measured snapshot cost: `MockSolver::snapshot`
//! is ~200 ms at 320x200 (per-cell area-Mach bisection), and publishing at
//! 16 ms against that would spend the whole thread inside snapshots. The real
//! solver's snapshot is a cheap conversion, so this adapts back to 16 ms by
//! itself at integration.

use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use cfd_contract::{Report, Snapshot, SolveSetup, Solver, SolverCommand, StepInfo};
use cfd_core::{EulerSolver, MockSolver};

/// Everything the UI reads per frame, published atomically.
#[derive(Clone)]
pub struct UiFrame {
    pub snapshot: Snapshot,
    pub report: Report,
    pub info: StepInfo,
    pub steps_per_sec: f64,
    /// Measured solver throughput of THIS machine in cells/s, from the short
    /// calibration run at each solver build. The cost readout's wall-clock
    /// estimate divides by this — never by a constant from another machine.
    /// 0.0 until the first calibration completes.
    pub cells_per_sec: f64,
    /// The worker's own pause state; diverges from the UI's only when the
    /// solver errored and paused itself.
    pub paused: bool,
    /// Set when the solver returned an error; the worker pauses itself.
    pub error: Option<String>,
}

pub enum UiCommand {
    Solver(SolverCommand),
    /// Full restart with new reference scales (chamber-pressure change).
    /// `RefScales` is fixed at solver construction, so this is the one edit
    /// that genuinely requires discarding the field.
    Rebuild(Box<SolveSetup>),
    Quit,
}

/// Minimum wall-clock between publishes. Load-bearing: without it a fast
/// solver fires `request_repaint` hundreds of times a second and the UI
/// thread never gets scheduled.
const MIN_PUBLISH: Duration = Duration::from_millis(16);
/// Per-window frame-time budget for stepping (the turbo gate).
const STEP_BUDGET: Duration = Duration::from_millis(12);
/// Steps per second at 1x. One step per 16 ms rendered frame.
const BASE_STEP_RATE: f64 = 60.0;

const FRESH_INFO: StepInfo = StepInfo {
    step: 0,
    time: 0.0,
    dt: 0.0,
    residual: f64::NAN,
    converged: false,
    floor_activations: 0,
};

/// Which solver to build. Real by default; `CFD_SOLVER=mock` flips back to
/// the analytic preview without a rebuild (abort-ladder rung 0).
pub fn solver_kind() -> cfd_contract::SolverKind {
    match std::env::var("CFD_SOLVER").as_deref() {
        Ok("mock") => cfd_contract::SolverKind::Mock,
        _ => cfd_contract::SolverKind::Real,
    }
}

pub fn build(setup: &SolveSetup) -> cfd_contract::Result<Box<dyn Solver>> {
    // Allocation guard: refuse a grid whose solver buffers cannot fit in the
    // memory the machine has to give, with a message instead of an OOM abort.
    // The UI's blocking confirmation is the first line of defence; this is
    // the second (the estimate is exactly that — an estimate).
    let need = setup.grid.glen() as u64 * crate::case::EST_SOLVER_BYTES_PER_CELL;
    let have = crate::case::memory_budget_bytes();
    if need > have {
        return Err(cfd_contract::CfdError::Parameter(format!(
            "grid of {} x {} cells needs ~{} MB of solver buffers but only \
             ~{} MB of memory is available — reduce the domain or resolution",
            setup.grid.nz,
            setup.grid.nr,
            need / (1024 * 1024),
            have / (1024 * 1024)
        )));
    }
    Ok(match solver_kind() {
        cfd_contract::SolverKind::Mock => Box::new(MockSolver::new(setup.clone())?),
        cfd_contract::SolverKind::Real => Box::new(EulerSolver::new(setup.clone())?),
    })
}

/// Short timed run measuring this machine's solver throughput in cells/s,
/// used by the sidebar cost readout. Runs right after every build, on the
/// startup transient the solver would be stepping through anyway; capped at
/// ~300 ms so a rebuild never stalls noticeably. Returns the throughput and
/// the last step's info (the steps are real and count normally).
fn calibrate(solver: &mut dyn Solver, info: &mut StepInfo, error: &mut Option<String>) -> f64 {
    let cells = solver.snapshot().grid.len();
    let t0 = Instant::now();
    let mut n = 0u32;
    while n < 24 && t0.elapsed() < Duration::from_millis(300) {
        match solver.step() {
            Ok(i) => {
                *info = i;
                n += 1;
            }
            Err(e) => {
                *error = Some(e.to_string());
                break;
            }
        }
    }
    let el = t0.elapsed().as_secs_f64();
    if n == 0 || el <= 0.0 {
        0.0
    } else {
        n as f64 * cells as f64 / el
    }
}

pub fn make_frame(solver: &dyn Solver, info: StepInfo) -> UiFrame {
    UiFrame {
        snapshot: solver.snapshot(),
        report: solver.report(),
        info,
        steps_per_sec: 0.0,
        cells_per_sec: 0.0,
        paused: false,
        error: None,
    }
}

pub fn spawn(
    mut setup: SolveSetup,
    mut solver: Box<dyn Solver>,
    mut info: StepInfo,
    ctx: eframe::egui::Context,
    rx: Receiver<UiCommand>,
    mut tx: triple_buffer::Input<UiFrame>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("cfd-solver".into())
        .spawn(move || {
            let mut paused = false;
            let mut turbo: u32 = 1;
            let mut error: Option<String> = None;
            let mut publish_interval = MIN_PUBLISH;
            let mut last_publish = Instant::now() - publish_interval;
            let mut dirty = true; // publish the initial state once
                                  // Wall-clock step pacing.
            let mut last_pace = Instant::now();
            let mut step_debt = 0.0f64;
            // Rolling steps/s.
            let mut rate_t0 = Instant::now();
            let mut rate_steps: u64 = 0;
            let mut steps_per_sec = 0.0f64;
            // Throughput calibration for the cost readout, re-measured at
            // every build so it always describes the current grid on this
            // machine.
            let mut cells_per_sec = calibrate(solver.as_mut(), &mut info, &mut error);

            'main: loop {
                let mut single_step = false;
                while let Ok(cmd) = rx.try_recv() {
                    match cmd {
                        UiCommand::Quit => break 'main,
                        UiCommand::Rebuild(new_setup) => {
                            setup = *new_setup;
                            match build(&setup) {
                                Ok(s) => {
                                    solver = s;
                                    info = FRESH_INFO;
                                    error = None;
                                    cells_per_sec =
                                        calibrate(solver.as_mut(), &mut info, &mut error);
                                }
                                Err(e) => error = Some(e.to_string()),
                            }
                        }
                        UiCommand::Solver(cmd) => match cmd {
                            SolverCommand::Pause(p) => {
                                paused = p;
                                step_debt = 0.0;
                                last_pace = Instant::now();
                            }
                            SolverCommand::SingleStep => single_step = true,
                            SolverCommand::Turbo(t) => turbo = t.max(1),
                            SolverCommand::Reset => match build(&setup) {
                                Ok(s) => {
                                    solver = s;
                                    info = FRESH_INFO;
                                    error = None;
                                    cells_per_sec =
                                        calibrate(solver.as_mut(), &mut info, &mut error);
                                }
                                Err(e) => error = Some(e.to_string()),
                            },
                            SolverCommand::SetAmbient(a) => {
                                setup.ambient = a;
                                solver.set_ambient(a);
                            }
                            SolverCommand::SetGeometry(solid) => {
                                setup.solid = solid.clone();
                                if let Err(e) = solver.set_geometry(solid) {
                                    error = Some(e.to_string());
                                }
                            }
                            SolverCommand::SetNumerics(n) => {
                                setup.numerics = n;
                                solver.set_numerics(n);
                            }
                        },
                    }
                    dirty = true;
                }

                let idle = (paused && !single_step) || error.is_some();
                if !idle {
                    // Steps due by wall clock since the last pacing mark.
                    let now = Instant::now();
                    if single_step {
                        step_debt = 1.0;
                    } else {
                        step_debt += now.duration_since(last_pace).as_secs_f64()
                            * BASE_STEP_RATE
                            * turbo as f64;
                        // Allow catching up across one long snapshot+publish
                        // cycle (so a slow snapshot never slows simulated
                        // time), but never a runaway burst after a stall.
                        let cap =
                            (2.0 * BASE_STEP_RATE * turbo as f64 * publish_interval.as_secs_f64())
                                .max(4.0 * turbo as f64);
                        step_debt = step_debt.min(cap);
                    }
                    last_pace = now;

                    let budget_t0 = Instant::now();
                    while step_debt >= 1.0 {
                        match solver.step() {
                            Ok(i) => {
                                info = i;
                                rate_steps += 1;
                                step_debt -= 1.0;
                            }
                            Err(e) => {
                                error = Some(e.to_string());
                                paused = true;
                                break;
                            }
                        }
                        if budget_t0.elapsed() >= STEP_BUDGET {
                            step_debt = 0.0; // budget-gated: drop the backlog
                            break;
                        }
                        if single_step {
                            break;
                        }
                    }
                    dirty = true;
                }

                let el = rate_t0.elapsed();
                if el >= Duration::from_millis(500) {
                    steps_per_sec = rate_steps as f64 / el.as_secs_f64();
                    rate_steps = 0;
                    rate_t0 = Instant::now();
                }

                if dirty && last_publish.elapsed() >= publish_interval {
                    let t0 = Instant::now();
                    tx.write(UiFrame {
                        snapshot: solver.snapshot(),
                        report: solver.report(),
                        info,
                        steps_per_sec: if paused { 0.0 } else { steps_per_sec },
                        cells_per_sec,
                        paused,
                        error: error.clone(),
                    });
                    ctx.request_repaint();
                    // Adapt to the snapshot cost so the thread is never more
                    // than ~60% snapshot-bound.
                    let cost = t0.elapsed();
                    publish_interval =
                        (cost + cost / 2).clamp(MIN_PUBLISH, Duration::from_millis(400));
                    last_publish = Instant::now();
                    dirty = false;
                }

                std::thread::sleep(Duration::from_millis(if idle { 8 } else { 4 }));
            }
        })
        .expect("failed to spawn solver thread")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{self, CaseParams};
    use std::sync::Arc;
    use std::time::Duration;

    /// Poll the buffer until `pred` holds or the timeout expires; returns the
    /// last frame either way. The worker's publish interval adapts to the
    /// mock's ~200 ms snapshot cost, so fixed sleeps would race it.
    fn wait_for(
        out: &mut triple_buffer::Output<UiFrame>,
        timeout: Duration,
        pred: impl Fn(&UiFrame) -> bool,
    ) -> UiFrame {
        let t0 = Instant::now();
        loop {
            let f = out.read().clone();
            if pred(&f) || t0.elapsed() > timeout {
                return f;
            }
            std::thread::sleep(Duration::from_millis(40));
        }
    }

    const T: Duration = Duration::from_secs(5);

    /// Every control the UI exposes, exercised headlessly against the worker
    /// thread. An orphan `egui::Context` swallows the repaint requests.
    ///
    /// Pinned to the mock: this test checks the worker's command plumbing, and
    /// its assertions (e.g. the report tracking p0 across a Rebuild within
    /// seconds) rely on the mock's instant analytic response. The real solver
    /// needs convergence time and reports NaN until flow reaches the exit.
    #[test]
    fn worker_honours_every_command() {
        std::env::set_var("CFD_SOLVER", "mock");
        // Preview extents: this test checks command plumbing, and the mock's
        // per-cell snapshot cost on the Large default would only slow it.
        let (lz_rt, lr_rt, cells_per_rt, dz_over_dr) = case::DomainPreset::Preview.values();
        let params = CaseParams { lz_rt, lr_rt, cells_per_rt, dz_over_dr, ..CaseParams::default() };
        let wall = case::conical_contour(params.area_ratio);
        let setup = case::make_setup(&params, &wall);
        let solver_grid = setup.grid.clone();
        let solver: Box<dyn Solver> = Box::new(MockSolver::new(setup.clone()).unwrap());
        let initial = make_frame(solver.as_ref(), FRESH_INFO);
        let (buf_in, mut out) = triple_buffer::triple_buffer(&initial);
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = spawn(
            setup,
            solver,
            FRESH_INFO,
            eframe::egui::Context::default(),
            rx,
            buf_in,
        );
        let send = |c: SolverCommand| tx.send(UiCommand::Solver(c)).unwrap();

        // Runs on its own from the start.
        let f_run = wait_for(&mut out, T, |f| f.info.step > 5);
        assert!(
            f_run.info.step > 5,
            "solver did not advance: {}",
            f_run.info.step
        );

        // Pause stops stepping; SingleStep advances by exactly one.
        send(SolverCommand::Pause(true));
        let p1 = wait_for(&mut out, T, |f| f.paused);
        assert!(p1.paused, "pause never reached the worker");
        std::thread::sleep(Duration::from_millis(500));
        let p2 = out.read().clone();
        assert_eq!(p1.info.step, p2.info.step, "paused solver still stepping");
        send(SolverCommand::SingleStep);
        let p3 = wait_for(&mut out, T, |f| f.info.step == p2.info.step + 1);
        assert_eq!(p3.info.step, p2.info.step + 1, "single step was not single");

        // The altitude slider path: SetAmbient must NOT reset the run, and
        // the exit pressure ratio must move (30 km: p_a drops ~85x).
        let high = CaseParams {
            altitude_m: 30_000.0,
            ..params
        };
        send(SolverCommand::SetAmbient(case::ambient_nd(&high)));
        send(SolverCommand::Pause(false));
        let base_pr = f_run.report.exit_pressure_ratio;
        let f_high = wait_for(&mut out, T, |f| {
            f.report.exit_pressure_ratio > base_pr * 5.0
        });
        assert!(f_high.info.step >= p3.info.step, "SetAmbient reset the run");
        assert!(
            f_high.report.exit_pressure_ratio > base_pr * 5.0,
            "p_e/p_a did not rise at altitude: {} -> {}",
            base_pr,
            f_high.report.exit_pressure_ratio
        );

        // Turbo multiplies the wall-clock step rate.
        send(SolverCommand::Turbo(16));
        std::thread::sleep(Duration::from_millis(500));
        let a = out.read().info.step;
        std::thread::sleep(Duration::from_millis(1000));
        let b = out.read().info.step;
        send(SolverCommand::Turbo(1));
        std::thread::sleep(Duration::from_millis(500));
        let c = out.read().info.step;
        std::thread::sleep(Duration::from_millis(1000));
        let d = out.read().info.step;
        assert!(
            (b - a) as f64 > 3.0 * ((d - c).max(1) as f64),
            "turbo 16x not faster than 1x: {} vs {}",
            b - a,
            d - c
        );

        // Geometry edits reach the solver: a bigger bell raises exit Mach.
        let wall12 = case::conical_contour(12.0);
        // Mid-run geometry edits target the solver's CURRENT grid.
        let solid = case::rasterize_wall(&wall12, &solver_grid);
        let mach_before = f_high.report.exit_mach;
        send(SolverCommand::SetGeometry(Arc::new(solid)));
        let f_geo = wait_for(&mut out, T, |f| f.report.exit_mach > mach_before + 0.1);
        assert!(
            f_geo.report.exit_mach > mach_before + 0.1,
            "exit Mach did not rise with area ratio: {} -> {}",
            mach_before,
            f_geo.report.exit_mach
        );

        // Reset restarts the step count.
        let step_before = out.read().info.step;
        send(SolverCommand::Reset);
        let f_reset = wait_for(&mut out, T, |f| f.info.step < step_before);
        assert!(f_reset.info.step < step_before, "reset did not restart");

        // Rebuild (chamber pressure change) swaps the reference scales:
        // mass flow scales with p0.
        let low = CaseParams {
            p0_pa: 2.0e6,
            ..params
        };
        tx.send(UiCommand::Rebuild(Box::new(case::make_setup(&low, &wall))))
            .unwrap();
        let base_mdot = f_run.report.mass_flow_kg_s;
        let f_low = wait_for(&mut out, T, |f| f.report.mass_flow_kg_s < base_mdot * 0.6);
        assert!(
            f_low.report.mass_flow_kg_s < base_mdot * 0.6,
            "mass flow did not track p0: {} -> {}",
            base_mdot,
            f_low.report.mass_flow_kg_s
        );

        tx.send(UiCommand::Quit).unwrap();
        handle.join().unwrap();
    }
}
