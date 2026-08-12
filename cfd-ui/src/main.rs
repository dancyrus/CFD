//! The app. Session D owns this crate — sole owner of eframe/egui.
//! See docs/sessions/D-ui.md and docs/contract.md.
//!
//! Wiring (D-ui.md §1): the solver lives on its own thread; `Snapshot`s flow
//! out through a `triple_buffer` (latest wins, never blocks either side),
//! `SolverCommand`s flow in over `std::sync::mpsc`, and the worker calls
//! `ctx.request_repaint()` after each publish, throttled to 16 ms.

mod app;
mod canvas;
mod case;
mod colormap;
mod editor;
mod worker;

use cfd_contract::{Solver, StepInfo};
use cfd_core::MockSolver;

fn main() -> eframe::Result {
    let params = case::CaseParams::default();
    let wall = case::conical_contour(params.area_ratio);
    let setup = case::make_setup(&params, &wall);

    // MockSolver from minute one; EulerSolver is a one-line swap here and in
    // worker::build once sessions A and B land.
    let solver: Box<dyn Solver> =
        Box::new(MockSolver::new(setup.clone()).expect("mock solver rejected the demo case"));
    let info = StepInfo {
        step: 0,
        time: 0.0,
        dt: 0.0,
        residual: f64::NAN,
        converged: false,
        floor_activations: 0,
    };
    let initial = worker::make_frame(solver.as_ref(), info);

    let (buf_in, buf_out) = triple_buffer::triple_buffer(&initial);
    let (tx, rx) = std::sync::mpsc::channel::<worker::UiCommand>();
    let tx_exit = tx.clone();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 880.0])
            .with_min_inner_size([980.0, 600.0])
            .with_title("CFD Nozzle Sandbox — analytic preview"),
        vsync: true,
        ..Default::default()
    };

    eframe::run_native(
        "cfd-sandbox",
        options,
        Box::new(move |cc| {
            worker::spawn(setup, solver, info, cc.egui_ctx.clone(), rx, buf_in);
            Ok(Box::new(ExitGuard {
                app: app::CfdApp::new(buf_out, tx, initial, params, wall),
                tx: tx_exit,
            }))
        }),
    )
}

/// Wraps the app so the solver thread is told to quit when the window closes.
struct ExitGuard {
    app: app::CfdApp,
    tx: std::sync::mpsc::Sender<worker::UiCommand>,
}

impl eframe::App for ExitGuard {
    fn update(&mut self, ctx: &eframe::egui::Context, frame: &mut eframe::Frame) {
        self.app.update(ctx, frame);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let _ = self.tx.send(worker::UiCommand::Quit);
    }
}
