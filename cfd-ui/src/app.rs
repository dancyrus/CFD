//! The application: layout, controls, keybindings and the honesty surface.
//! docs/sessions/D-ui.md §3–§4 are the authority on what may and may not be
//! shown; docs/physics-reference.md §13 on the wording.

use std::sync::mpsc::Sender;
use std::sync::Arc;

use eframe::egui::{
    self, Align2, Color32, FontId, Key, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, Vec2,
};

use cfd_contract::{FieldKind, SolverCommand};

use crate::canvas::{colorbar, fmt_value, Canvas, BG};
use crate::case::{
    self, ambient_nd, atmosphere, conical_contour, ideal_cf, make_setup, rasterize_wall,
    separation_altitude_m, separation_threshold, CaseParams, ALT_MAX_M,
};
use crate::editor::{EditorBackend, StubEditor};
use crate::worker::{UiCommand, UiFrame};

const TURBO_STEPS: [u32; 3] = [1, 4, 16];
const AMBER: Color32 = Color32::from_rgb(235, 170, 45);
const RED: Color32 = Color32::from_rgb(235, 90, 80);
const GREEN: Color32 = Color32::from_rgb(90, 200, 120);

/// Selector order = keys 1..8. Schlieren first per the brief.
const FIELD_KEYS: [(Key, FieldKind, &str); 8] = [
    (Key::Num1, FieldKind::Schlieren, "Schlieren"),
    (Key::Num2, FieldKind::Mach, "Mach"),
    (Key::Num3, FieldKind::Pressure, "Pressure"),
    (Key::Num4, FieldKind::Temperature, "Temperature"),
    (Key::Num5, FieldKind::Density, "Density"),
    (Key::Num6, FieldKind::Speed, "Speed"),
    (Key::Num7, FieldKind::VelocityZ, "Velocity z"),
    (Key::Num8, FieldKind::VelocityR, "Velocity r"),
];

pub struct CfdApp {
    out: triple_buffer::Output<UiFrame>,
    tx: Sender<UiCommand>,
    latest: UiFrame,
    frame_gen: u64,

    params: CaseParams,
    // Slider staging (committed on release).
    ui_area_ratio: f64,
    ui_p0_mpa: f64,

    field: FieldKind,
    paused: bool,
    turbo_idx: usize,
    /// Locked per-field display ranges; refit is the only thing that moves
    /// them. An auto-rescaling colorbar makes two frames incomparable.
    locked: [(f32, f32); 8],
    smooth: bool,

    canvas: Canvas,
    editor: StubEditor,
    editor_on: bool,
    committed_wall: Vec<[f64; 2]>,
    /// True once the user has edited the wall by hand — flips the whole
    /// report into "sandbox — qualitative only".
    geometry_custom: bool,
    space_panned: bool,
    hover_text: String,
    /// ANALYTIC PREVIEW overlay — true only for MockSolver builds.
    watermark: bool,
}

impl CfdApp {
    pub fn new(
        out: triple_buffer::Output<UiFrame>,
        tx: Sender<UiCommand>,
        initial: UiFrame,
        params: CaseParams,
        wall: Vec<[f64; 2]>,
        watermark: bool,
    ) -> Self {
        let mut locked = [(0.0f32, 1.0f32); 8];
        for k in FieldKind::ALL {
            locked[k as usize] = lock_range(k, initial.snapshot.range(k));
        }
        CfdApp {
            out,
            tx,
            latest: initial,
            frame_gen: 0,
            ui_area_ratio: params.area_ratio,
            ui_p0_mpa: params.p0_pa / 1e6,
            params,
            field: FieldKind::Mach,
            paused: false,
            turbo_idx: 0,
            locked,
            smooth: false,
            canvas: Canvas::new(),
            editor: StubEditor::new(wall.clone()),
            editor_on: false,
            committed_wall: wall,
            geometry_custom: false,
            space_panned: false,
            hover_text: String::new(),
            watermark,
        }
    }

    fn cmd(&self, c: SolverCommand) {
        let _ = self.tx.send(UiCommand::Solver(c));
    }

    fn set_paused(&mut self, p: bool) {
        self.paused = p;
        self.cmd(SolverCommand::Pause(p));
    }

    fn commit_editor_geometry(&mut self) {
        self.committed_wall = self.editor.points().to_vec();
        self.geometry_custom = true;
        let solid = rasterize_wall(&self.committed_wall, &case::grid());
        self.cmd(SolverCommand::SetGeometry(Arc::new(solid)));
    }

    /// Parametric regeneration from the area-ratio slider: replaces any hand
    /// edits and clears sandbox mode.
    fn regenerate_contour(&mut self) {
        self.params.area_ratio = self.ui_area_ratio;
        let wall = conical_contour(self.params.area_ratio);
        self.editor.set_points(wall.clone());
        self.committed_wall = wall;
        self.geometry_custom = false;
        let solid = rasterize_wall(&self.committed_wall, &case::grid());
        self.cmd(SolverCommand::SetGeometry(Arc::new(solid)));
    }

    /// Chamber-pressure change: `RefScales` are fixed at construction, so this
    /// is the one control that rebuilds the solver and discards the field.
    fn rebuild_solver(&mut self) {
        self.params.p0_pa = self.ui_p0_mpa * 1e6;
        let setup = make_setup(&self.params, &self.committed_wall);
        let _ = self.tx.send(UiCommand::Rebuild(Box::new(setup)));
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        let (pressed, space_released) = ctx.input(|i| {
            let mut v: Vec<Key> = Vec::new();
            for k in [Key::Period, Key::R, Key::F, Key::T, Key::E, Key::L] {
                if i.key_pressed(k) {
                    v.push(k);
                }
            }
            for (k, _, _) in FIELD_KEYS {
                if i.key_pressed(k) {
                    v.push(k);
                }
            }
            (v, i.key_released(Key::Space))
        });
        // Space toggles play/pause unless the press was spent panning.
        if space_released {
            if !self.space_panned {
                let p = !self.paused;
                self.set_paused(p);
            }
            self.space_panned = false;
        }
        for k in pressed {
            match k {
                Key::Period => {
                    if !self.paused {
                        self.set_paused(true);
                    }
                    self.cmd(SolverCommand::SingleStep);
                }
                Key::R => self.cmd(SolverCommand::Reset),
                Key::F => self.canvas.request_fit(),
                Key::T => {
                    self.turbo_idx = (self.turbo_idx + 1) % TURBO_STEPS.len();
                    self.cmd(SolverCommand::Turbo(TURBO_STEPS[self.turbo_idx]));
                }
                Key::E => self.editor_on = !self.editor_on,
                Key::L => self.smooth = !self.smooth,
                k => {
                    if let Some((_, f, _)) = FIELD_KEYS.iter().find(|(fk, _, _)| *fk == k) {
                        self.field = *f;
                    }
                }
            }
        }
    }

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let play_label = if self.paused { "▶ Play" } else { "⏸ Pause" };
            if ui.button(play_label).on_hover_text("Space").clicked() {
                let p = !self.paused;
                self.set_paused(p);
            }
            if ui.button("⏭ Step").on_hover_text("Period (.)").clicked() {
                if !self.paused {
                    self.set_paused(true);
                }
                self.cmd(SolverCommand::SingleStep);
            }
            if ui.button("↺ Reset").on_hover_text("R").clicked() {
                self.cmd(SolverCommand::Reset);
            }
            ui.separator();
            ui.label("Turbo");
            for (i, t) in TURBO_STEPS.iter().enumerate() {
                if ui
                    .selectable_label(self.turbo_idx == i, format!("{t}×"))
                    .on_hover_text("T cycles")
                    .clicked()
                {
                    self.turbo_idx = i;
                    self.cmd(SolverCommand::Turbo(*t));
                }
            }
            ui.separator();
            ui.toggle_value(&mut self.editor_on, "✏ Edit walls")
                .on_hover_text("E — drag points, Ctrl+click to insert, right-click to remove");
            if ui.button("⛶ Fit").on_hover_text("F").clicked() {
                self.canvas.request_fit();
            }
            ui.separator();
            let (dot, label) = if self.latest.error.is_some() {
                (RED, "error")
            } else if self.paused {
                (AMBER, "paused")
            } else if self.latest.info.converged {
                (GREEN, "settled")
            } else {
                (Color32::from_rgb(110, 170, 255), "converging")
            };
            let (r, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
            ui.painter().circle_filled(r.center(), 4.0, dot);
            ui.label(RichText::new(label).weak());
        });
    }

    fn display_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Display");
        ui.horizontal_wrapped(|ui| {
            for (i, (_, f, name)) in FIELD_KEYS.iter().enumerate() {
                if ui
                    .selectable_label(self.field == *f, format!("{name} [{}]", i + 1))
                    .clicked()
                {
                    self.field = *f;
                }
            }
        });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            colorbar(ui, self.field, self.locked[self.field as usize], 120.0);
            ui.vertical(|ui| {
                ui.label(RichText::new(self.field.label()).strong());
                ui.label(RichText::new("range locked").weak().small());
                if ui
                    .button("Refit range")
                    .on_hover_text(
                        "Re-lock the colorbar to the current field's min/max. \
                         The range never rescales on its own — an auto-scaling \
                         colorbar makes two frames incomparable.",
                    )
                    .clicked()
                {
                    self.locked[self.field as usize] =
                        lock_range(self.field, self.latest.snapshot.range(self.field));
                }
                ui.checkbox(&mut self.smooth, "Smooth (L)")
                    .on_hover_text("LINEAR filtering; default NEAREST shows cells");
            });
        });
    }

    fn operating_point(&mut self, ui: &mut egui::Ui) {
        ui.heading("Operating point");

        // ---- Altitude: never resets the field; the transient IS the demo.
        let sep_m = separation_altitude_m(&self.params);
        ui.label("Altitude");
        if altitude_slider(ui, &mut self.params.altitude_m, sep_m) {
            self.cmd(SolverCommand::SetAmbient(ambient_nd(&self.params)));
        }
        let (pa, ta) = atmosphere(self.params.altitude_m);
        ui.label(
            RichText::new(format!(
                "{:.1} km · p∞ {} · T∞ {:.0} K",
                self.params.altitude_m / 1000.0,
                fmt_pressure(pa),
                ta
            ))
            .weak()
            .small(),
        );
        ui.add_space(6.0);

        // ---- Area ratio (regenerates the parametric contour on release).
        let r = ui.add(
            egui::Slider::new(&mut self.ui_area_ratio, 2.0..=16.0)
                .text("Area ratio ε")
                .fixed_decimals(1),
        );
        if r.drag_stopped() || (r.changed() && !r.dragged()) {
            self.regenerate_contour();
        }

        // ---- Chamber pressure (the one field-discarding control).
        let r = ui.add(
            egui::Slider::new(&mut self.ui_p0_mpa, 0.5..=10.0)
                .text("Chamber p₀ [MPa]")
                .fixed_decimals(1),
        );
        if r.drag_stopped() || (r.changed() && !r.dragged()) {
            self.rebuild_solver();
        }
        ui.label(
            RichText::new("p₀ changes the reference scales: rebuilds and restarts the field.")
                .weak()
                .small(),
        );
    }

    fn report_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Report");
        let rep = &self.latest.report;
        let info = &self.latest.info;

        if info.floor_activations > 0 {
            // The product rule: nothing displays while the floor counter is
            // nonzero. Every downstream number is un-auditable.
            ui.colored_label(RED, RichText::new("SOLUTION INVALID").strong());
            ui.label(format!(
                "Positivity floor activated {} time(s). No quantitative output \
                 will be shown for this field.",
                info.floor_activations
            ));
            return;
        }

        if self.geometry_custom {
            ui.colored_label(
                AMBER,
                RichText::new("SANDBOX — drawn geometry, qualitative only").small(),
            );
            ui.label(
                RichText::new("No acceptance test covers an arbitrary user drawing.")
                    .weak()
                    .small(),
            );
        }

        let settled = rep.converged;
        if !settled {
            ui.label(
                RichText::new("settling — integrated quantities greyed until the residual drops")
                    .weak()
                    .small(),
            );
        }
        // Rounded before the threshold test: the detected throat radius is
        // quantized to a cell face, and 1.0/dr in f32 lands at 19.9999997.
        let n_throat = rep.cells_per_throat_radius.round();
        let underresolved = n_throat.is_finite() && n_throat < 20.0;

        // Greyed-out whenever unsettled: a report from an unconverged field
        // is not a number.
        ui.add_enabled_ui(settled, |ui| {
            egui::Grid::new("report")
                .num_columns(3)
                .spacing([10.0, 3.0])
                .show(ui, |ui| {
                    ui.label("mass flow");
                    ui.monospace(fmt_or_dash(rep.mass_flow_kg_s, "kg/s"));
                    ui.label(RichText::new("±5%").small().color(if underresolved {
                        AMBER
                    } else {
                        ui.visuals().weak_text_color()
                    }));
                    ui.end_row();

                    ui.label("thrust");
                    ui.monospace(if rep.thrust_n.is_finite() {
                        format!("{:.2} kN", rep.thrust_n / 1000.0)
                    } else {
                        "—".into()
                    });
                    // N_throat lives NEXT TO the thrust readout, by decree.
                    // The bias figure is measured (T8, ladder.rs): the
                    // staircase wall's entropy layer costs ~13% of thrust at
                    // the default N_throat = 20.
                    ui.label(
                        RichText::new(format!(
                            "{n_throat:.0} cells / r_t · ≈13% low (staircase wall)"
                        ))
                        .small()
                        .color(if underresolved {
                            AMBER
                        } else {
                            ui.visuals().weak_text_color()
                        }),
                    );
                    ui.end_row();

                    let pa_p0 = ambient_nd(&self.params).p as f64;
                    let cf_ideal = ideal_cf(self.params.area_ratio, self.params.gamma, pa_p0);
                    ui.label("C_f");
                    ui.monospace(fmt_or_dash(rep.thrust_coefficient, ""));
                    ui.label(
                        RichText::new(format!(
                            "{:.3} of 1-D ideal",
                            rep.thrust_coefficient / cf_ideal
                        ))
                        .small()
                        .weak(),
                    );
                    ui.end_row();

                    ui.label("c*");
                    ui.monospace(fmt_or_dash(rep.c_star_m_s, "m/s"));
                    ui.label(RichText::new("").small());
                    ui.end_row();

                    ui.label("C_d");
                    ui.monospace(fmt_or_dash(rep.discharge_coefficient, ""));
                    ui.label(RichText::new("").small());
                    ui.end_row();

                    ui.label("exit Mach");
                    ui.monospace(fmt_or_dash(rep.exit_mach, ""));
                    // Measured (T8): the wall layer drags the area average
                    // ~19% below the 1-D ideal at default resolution; the
                    // core flow is within ~7%.
                    ui.label(
                        RichText::new(format!(
                            "area-avg · ≈19% low at this res (staircase wall) · 1-D ideal {:.2}",
                            rep.ideal_exit_mach
                        ))
                        .small()
                        .weak(),
                    );
                    ui.end_row();

                    ui.label("p_e / p_a");
                    ui.monospace(fmt_or_dash(rep.exit_pressure_ratio, ""));
                    ui.label(RichText::new("area-averaged ±4%").small().weak());
                    ui.end_row();
                });
            ui.label(
                RichText::new(
                    "C_f is inviscid, no divergence correction applied. \
                               Isp is not shown: it needs real-gas chemistry this \
                               model does not have.",
                )
                .weak()
                .small(),
            );
        });
    }

    fn residual_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Residual");
        let res = self.latest.info.residual;
        let (rect, _) = ui.allocate_exact_size(
            Vec2::new(ui.available_width().min(260.0), 14.0),
            Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, ui.visuals().extreme_bg_color);
        if res.is_finite() && res > 0.0 {
            // log10 in [-4, 0] -> bar fill, right-to-left convergence.
            let t = ((res.log10() + 4.0) / 4.0).clamp(0.0, 1.0) as f32;
            let fill = Rect::from_min_size(
                rect.min,
                Vec2::new(rect.width() * t.max(0.02), rect.height()),
            );
            let color = if res < 1e-3 { GREEN } else { AMBER };
            painter.rect_filled(fill, 3.0, color.gamma_multiply(0.65));
            // Threshold tick at 1e-3.
            let x = rect.left() + rect.width() * 0.25;
            painter.line_segment(
                [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                Stroke::new(1.0, ui.visuals().weak_text_color()),
            );
        }
        painter.rect_stroke(
            rect,
            3.0,
            Stroke::new(1.0, Color32::from_gray(80)),
            StrokeKind::Outside,
        );
        ui.horizontal(|ui| {
            let (dot, txt) = if !res.is_finite() {
                (
                    ui.visuals().weak_text_color(),
                    "warming up (residual defined from step 10)".to_string(),
                )
            } else if res < 1e-3 {
                (GREEN, format!("settled · L2(∂ρ/∂t) = {res:.1e}"))
            } else {
                (AMBER, format!("converging · L2(∂ρ/∂t) = {res:.1e}"))
            };
            let (r, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
            ui.painter().circle_filled(r.center(), 4.0, dot);
            ui.label(RichText::new(txt).small());
        });
    }

    fn warnings_section(&mut self, ui: &mut egui::Ui) {
        if let Some(err) = &self.latest.error {
            warning_box(
                ui,
                RED,
                "SOLVER ERROR",
                &format!("{err}. The solver is paused; Reset (R) restarts it."),
            );
        }
        let rep = &self.latest.report;
        if rep.exit_pressure_ratio.is_finite() && rep.exit_mach.is_finite() {
            let thr = separation_threshold(rep.exit_mach);
            if rep.exit_pressure_ratio < thr {
                // Wording fixed by docs/physics-reference.md §13.
                warning_box(
                    ui,
                    RED,
                    "⚠ SEPARATED FLOW — NOT SIMULATED",
                    &format!(
                        "Exit pressure ratio p_e/p_a = {:.2} is below the separation \
                         threshold of {:.2}. A real nozzle would separate inside the \
                         divergent section: the boundary layer would detach and a shock \
                         would stand inside the nozzle. This simulation is inviscid — \
                         it has no boundary layer and cannot separate. Thrust and \
                         exit-pressure readouts are not valid in this regime.",
                        rep.exit_pressure_ratio, thr
                    ),
                );
            }
        }
    }

    fn bottom_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // The one permanent line of screen space. Non-negotiable.
            ui.label(
                RichText::new(format!(
                    "Inviscid Euler · γ = {} · no boundary layer · no chemistry · no heat transfer",
                    self.params.gamma
                ))
                .small(),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let info = &self.latest.info;
                ui.label(
                    RichText::new(format!(
                        "step {} · t = {:.2} ms · {:.0} steps/s",
                        info.step,
                        self.latest.snapshot.time_s * 1e3,
                        self.latest.steps_per_sec,
                    ))
                    .weak()
                    .small()
                    .monospace(),
                );
                if !self.hover_text.is_empty() {
                    ui.separator();
                    ui.label(RichText::new(&self.hover_text).small().monospace());
                }
            });
        });
    }
}

impl eframe::App for CfdApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Pull the newest published frame; the clone only happens when the
        // worker actually published (max ~60/s), not per repaint.
        if self.out.updated() {
            self.latest = self.out.read().clone();
            self.frame_gen += 1;
            // The worker pauses itself on a solver error; mirror that here so
            // the transport bar tells the truth.
            if self.latest.error.is_some() && self.latest.paused {
                self.paused = true;
            }
        }

        self.handle_keys(ctx);

        egui::TopBottomPanel::top("top").show(ctx, |ui| self.top_bar(ui));
        egui::TopBottomPanel::bottom("bottom").show(ctx, |ui| self.bottom_bar(ui));

        egui::SidePanel::right("side")
            .resizable(true)
            .default_width(330.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.display_section(ui);
                    ui.separator();
                    self.operating_point(ui);
                    ui.separator();
                    self.report_section(ui);
                    ui.separator();
                    self.residual_section(ui);
                    ui.separator();
                    self.warnings_section(ui);
                    ui.add_space(8.0);
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame {
                fill: BG,
                ..Default::default()
            })
            .show(ctx, |ui| {
                // Disjoint field borrows: canvas and editor mutably, the
                // frame immutably — one expression, no clone.
                let output = self.canvas.show(
                    ui,
                    &self.latest,
                    self.frame_gen,
                    self.field,
                    self.locked[self.field as usize],
                    self.smooth,
                    &mut self.editor,
                    self.editor_on,
                    &self.committed_wall,
                    self.watermark, // ANALYTIC PREVIEW watermark: mock builds only
                );
                if output.space_panned {
                    self.space_panned = true;
                }
                if output.commit_geometry {
                    self.commit_editor_geometry();
                }
                self.hover_text = match &output.hover {
                    Some(h) => {
                        let l_m = self.params.r_throat_m;
                        let pos = format!(
                            "z {:7.1} mm  r {:+7.1} mm",
                            h.z_nd * l_m * 1e3,
                            h.r_nd * l_m * 1e3
                        );
                        match h.value {
                            Some(v) => format!("{pos}  ·  {} {}", self.field.label(), fmt_value(v)),
                            None => format!("{pos}  ·  wall"),
                        }
                    }
                    None => String::new(),
                };
            });
    }
}

fn lock_range(kind: FieldKind, r: (f32, f32)) -> (f32, f32) {
    match kind {
        // Schlieren arrives normalized to [0, 1] every frame.
        FieldKind::Schlieren => (0.0, 1.0),
        // Signed fields get a symmetric range so zero sits on the diverging
        // colormap's white point.
        k if k.is_signed() => {
            let m = r.0.abs().max(r.1.abs()).max(1e-6);
            (-m, m)
        }
        _ => {
            if r.1 > r.0 {
                r
            } else {
                (r.0, r.0 + 1.0)
            }
        }
    }
}

fn fmt_or_dash(v: f64, unit: &str) -> String {
    if v.is_finite() {
        let s = fmt_value(v as f32);
        if unit.is_empty() {
            s
        } else {
            format!("{s} {unit}")
        }
    } else {
        "—".into()
    }
}

fn fmt_pressure(pa: f64) -> String {
    if pa >= 10_000.0 {
        format!("{:.1} kPa", pa / 1000.0)
    } else {
        format!("{pa:.0} Pa")
    }
}

fn warning_box(ui: &mut egui::Ui, color: Color32, title: &str, body: &str) {
    egui::Frame {
        fill: color.gamma_multiply(0.12),
        stroke: Stroke::new(1.0, color),
        inner_margin: egui::Margin::same(8),
        corner_radius: egui::CornerRadius::same(4),
        ..Default::default()
    }
    .show(ui, |ui| {
        ui.colored_label(color, RichText::new(title).strong());
        ui.label(RichText::new(body).small());
    });
    ui.add_space(4.0);
}

/// Custom altitude slider: the track is shaded below the separation crossing
/// and carries a labeled tick there, recomputed as the area ratio moves, so
/// the user sees where the trustworthy range ends BEFORE dragging.
fn altitude_slider(ui: &mut egui::Ui, alt_m: &mut f64, sep_m: Option<f64>) -> bool {
    let width = ui.available_width().min(300.0);
    let (rect, response) = ui.allocate_exact_size(Vec2::new(width, 48.0), Sense::click_and_drag());
    let track_y = rect.top() + 26.0;
    let (x0, x1) = (rect.left() + 8.0, rect.right() - 8.0);
    let x_of = |m: f64| x0 + ((m / ALT_MAX_M) as f32) * (x1 - x0);

    let mut changed = false;
    if response.dragged() || response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let m = (((pos.x - x0) / (x1 - x0)) as f64 * ALT_MAX_M).clamp(0.0, ALT_MAX_M);
            if (m - *alt_m).abs() > 1.0 {
                *alt_m = m;
                changed = true;
            }
        }
    }

    let painter = ui.painter_at(rect);
    painter.line_segment(
        [Pos2::new(x0, track_y), Pos2::new(x1, track_y)],
        Stroke::new(4.0, ui.visuals().widgets.inactive.bg_fill),
    );
    // Separation zone: everything below the crossing altitude is where a real
    // nozzle would separate — shade it before the user drags into it.
    if let Some(sep) = sep_m {
        let xs = x_of(sep);
        painter.line_segment(
            [Pos2::new(x0, track_y), Pos2::new(xs, track_y)],
            Stroke::new(4.0, RED.gamma_multiply(0.55)),
        );
        painter.line_segment(
            [Pos2::new(xs, track_y - 7.0), Pos2::new(xs, track_y + 7.0)],
            Stroke::new(2.0, RED),
        );
        painter.text(
            Pos2::new(xs.clamp(x0 + 30.0, x1 - 30.0), track_y - 10.0),
            Align2::CENTER_BOTTOM,
            format!("separation below {:.1} km", sep / 1000.0),
            FontId::proportional(10.0),
            RED,
        );
    }
    // 10 km ticks.
    for km in (0..=40).step_by(10) {
        let x = x_of(km as f64 * 1000.0);
        painter.line_segment(
            [Pos2::new(x, track_y + 4.0), Pos2::new(x, track_y + 8.0)],
            Stroke::new(1.0, ui.visuals().weak_text_color()),
        );
        painter.text(
            Pos2::new(x, track_y + 9.0),
            Align2::CENTER_TOP,
            format!("{km}"),
            FontId::proportional(9.0),
            ui.visuals().weak_text_color(),
        );
    }
    // Handle.
    let hx = x_of(*alt_m);
    painter.circle(
        Pos2::new(hx, track_y),
        7.0,
        ui.visuals().widgets.active.bg_fill,
        Stroke::new(1.5, ui.visuals().widgets.active.fg_stroke.color),
    );
    changed
}
