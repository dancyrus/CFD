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

use cfd_contract::{Solver, SolverKind, StepInfo};

fn main() -> eframe::Result {
    let params = case::CaseParams::default();
    let wall = case::nozzle_contour(&params);
    let setup = case::make_setup(&params, &wall.points);

    // EulerSolver by default; CFD_SOLVER=mock flips back to the analytic
    // preview without a rebuild (abort-ladder rung 0).
    let mock = worker::solver_kind() == SolverKind::Mock;
    let solver: Box<dyn Solver> =
        worker::build(&setup).expect("solver rejected the demo case");
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
            .with_title(if mock {
                "CFD Nozzle Sandbox — analytic preview"
            } else {
                "CFD Nozzle Sandbox"
            }),
        vsync: true,
        ..Default::default()
    };

    eframe::run_native(
        "cfd-sandbox",
        options,
        Box::new(move |cc| {
            worker::spawn(setup, solver, info, cc.egui_ctx.clone(), rx, buf_in);
            Ok(Box::new(ExitGuard {
                app: app::CfdApp::new(buf_out, tx, initial, params, wall, mock, cc.storage),
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

    /// eframe saves through the outermost `App`, which is this wrapper, so the
    /// delegation is what actually persists the colormap choice.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        self.app.save(storage);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let _ = self.tx.send(worker::UiCommand::Quit);
    }
}
