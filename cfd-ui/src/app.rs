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
    self, ambient_nd, atmosphere, ideal_cf, make_setup, nozzle_contour, rasterize_wall,
    separation_altitude_m, separation_threshold, CaseParams, ContourKind, PlumeLength, ALT_MAX_M,
    PRESETS, R_UNIVERSAL_SI, VACUUM_P_FRAC,
};
use crate::editor::{EditorBackend, StubEditor};
use crate::worker::{UiCommand, UiFrame};

const TURBO_STEPS: [u32; 3] = [1, 4, 16];
/// Step of a rebuilt solver at which the deferred colorbar re-lock fires.
const RELOCK_STEP: u64 = 30;
/// Floor-activation quarantine (docs/physics-reference.md §13): the report
/// blanks while any activation is younger than this many steps, then shows
/// numbers with a permanent amber disclosure. The startup front of the
/// high-pressure presets brushes the 1e-6 floor by design (§5) — measured
/// worst case: last activation at step 796 (Merlin Vac, cold start into
/// vacuum), zero afterwards across all six presets. 1500 clean steps is
/// roughly twice the longest observed activation window and several domain
/// flush times, so the startup's invented mass has left the domain.
const FLOOR_QUARANTINE_STEPS: u64 = 1500;
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
    /// Index into `case::PRESETS`; `None` is the custom/demo case. Touching
    /// any slider or the wall editor clears it — a preset is all-or-nothing.
    preset: Option<usize>,
    /// Re-lock the colorbar ranges after a rebuild: a preset or p₀ change
    /// can move the reference scales by 35×, and ranges locked to the old
    /// scales display as a solid color. Two stages — `relock_pending` is set
    /// when the rebuild is sent, `relock_armed` when the new solver's frames
    /// arrive (step count restarted); the actual re-lock waits until
    /// `RELOCK_STEP` so it does not capture the near-quiescent init field.
    relock_pending: bool,
    relock_armed: bool,
    /// Floor-activation quarantine tracking: the highest counter value seen
    /// and the solver step at which it last grew. A counter decrease means a
    /// rebuilt/reset solver; a step decrease with an unchanged counter means
    /// `set_ambient` restarted the step count (new transient) — both re-arm
    /// the quarantine conservatively.
    floors_seen: u64,
    floor_last_step: u64,
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
    /// The contour kind the generator ACTUALLY produced for `committed_wall`,
    /// and the rejection message when it fell back to the cone. Never the
    /// requested kind: `params.contour_kind` is a request, and reading the
    /// status line off it printed "80% bell" over a fallback cone. `None` is
    /// a wall no generator produced — a hand-dragged one — which has no
    /// contour family and no wall angles to name.
    wall_kind: Option<ContourKind>,
    wall_fallback: Option<String>,
    /// True once the user has edited the wall by hand — flips the whole
    /// report into "sandbox — qualitative only".
    geometry_custom: bool,
    space_panned: bool,
    hover_text: String,
    /// Cached plume-option time-to-steady estimates and the frame generation
    /// they were computed at (each estimate re-runs the grading, so it is
    /// refreshed on a slow cadence rather than per repaint).
    plume_est: Option<([f64; 3], u64)>,
    /// ANALYTIC PREVIEW overlay — true only for MockSolver builds.
    watermark: bool,
}

impl CfdApp {
    pub fn new(
        out: triple_buffer::Output<UiFrame>,
        tx: Sender<UiCommand>,
        initial: UiFrame,
        params: CaseParams,
        wall: case::GeneratedWall,
        watermark: bool,
    ) -> Self {
        let mut locked = [(0.0f32, 1.0f32); 8];
        for k in FieldKind::ALL {
            locked[k as usize] = lock_range(k, initial.snapshot.range(k));
        }
        let mut editor = StubEditor::new(wall.points.clone());
        {
            let (lz, lr) = case::domain(&params);
            editor.set_domain(lz, lr);
        }
        CfdApp {
            out,
            tx,
            latest: initial,
            frame_gen: 0,
            ui_area_ratio: params.area_ratio,
            ui_p0_mpa: params.p0_pa / 1e6,
            params,
            preset: None,
            relock_pending: false,
            relock_armed: false,
            floors_seen: 0,
            floor_last_step: 0,
            field: FieldKind::Mach,
            paused: false,
            turbo_idx: 0,
            locked,
            smooth: false,
            canvas: Canvas::new(),
            editor,
            editor_on: false,
            committed_wall: wall.points,
            wall_kind: Some(wall.kind),
            wall_fallback: wall.fallback,
            geometry_custom: false,
            space_panned: false,
            hover_text: String::new(),
            plume_est: None,
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
        self.preset = None; // hand-drawn walls are no named engine
        // …and no contour family either. This is the fourth writer of
        // `committed_wall`; leaving the generator's provenance behind here
        // would put "measured bell · θ_n 32.0°" over a wall the user dragged
        // by hand, which is the same lie as naming a bell over a fallback cone.
        self.wall_kind = None;
        self.wall_fallback = None;
        // Mid-run edits rasterize onto the solver's CURRENT (graded) grid —
        // that is the cheap-to-overwrite-mask property the sandbox rests on.
        // The grading itself is only recomputed on a rebuild (preset, p0 or
        // plume-length change), where the field restarts anyway.
        let solid = rasterize_wall(&self.committed_wall, &self.latest.snapshot.grid);
        self.cmd(SolverCommand::SetGeometry(Arc::new(solid)));
    }

    /// Parametric regeneration from the area-ratio slider: replaces any hand
    /// edits and clears sandbox mode.
    fn regenerate_contour(&mut self) {
        if self.params.area_ratio != self.ui_area_ratio {
            self.preset = None; // partial change: no longer the named engine
        }
        self.params.area_ratio = self.ui_area_ratio;
        let wall = nozzle_contour(&self.params);
        self.editor.set_points(wall.points.clone());
        self.committed_wall = wall.points;
        self.wall_kind = Some(wall.kind);
        self.wall_fallback = wall.fallback;
        self.geometry_custom = false;
        let solid = rasterize_wall(&self.committed_wall, &self.latest.snapshot.grid);
        self.cmd(SolverCommand::SetGeometry(Arc::new(solid)));
    }

    /// Chamber-pressure change: `RefScales` are fixed at construction, so this
    /// control rebuilds the solver and discards the field.
    fn rebuild_solver(&mut self) {
        if self.params.p0_pa != self.ui_p0_mpa * 1e6 {
            self.preset = None; // partial change: no longer the named engine
        }
        self.params.p0_pa = self.ui_p0_mpa * 1e6;
        let setup = make_setup(&self.params, &self.committed_wall);
        let _ = self.tx.send(UiCommand::Rebuild(Box::new(setup)));
        self.relock_pending = true;
    }

    /// Apply a preset whole — geometry, area ratio, chamber pressure, gas
    /// model and domain together, never partially — or revert to the demo
    /// case (`None`). Altitude and vacuum mode carry over: they describe the
    /// ambient, not the engine.
    fn apply_preset(&mut self, preset: Option<usize>) {
        self.preset = preset;
        let (alt, vac) = (self.params.altitude_m, self.params.vacuum);
        self.params = match preset {
            Some(i) => PRESETS[i].case(alt, vac),
            None => CaseParams {
                altitude_m: alt,
                vacuum: vac,
                ..CaseParams::default()
            },
        };
        self.ui_area_ratio = self.params.area_ratio;
        self.ui_p0_mpa = self.params.p0_pa / 1e6;
        let wall = nozzle_contour(&self.params);
        let (lz, lr) = case::domain(&self.params);
        self.editor.set_domain(lz, lr);
        self.editor.set_points(wall.points.clone());
        self.committed_wall = wall.points;
        self.wall_kind = Some(wall.kind);
        self.wall_fallback = wall.fallback;
        self.geometry_custom = false;
        let setup = make_setup(&self.params, &self.committed_wall);
        let _ = self.tx.send(UiCommand::Rebuild(Box::new(setup)));
        self.relock_pending = true;
        self.canvas.request_fit(); // the domain just changed size
    }

    /// Plume length control (grid-grading work order, item 3): Compact /
    /// Standard / Long with an estimated time to steady state per option.
    /// Switching rebuilds the solver on the re-graded grid (the domain
    /// changes, so the field restarts).
    fn plume_selector(&mut self, ui: &mut egui::Ui) {
        // Refresh the estimates at ~1 Hz: each one re-runs the grading and
        // rasterization for its candidate domain.
        let stale = match self.plume_est {
            None => true,
            Some((_, gen)) => self.frame_gen.saturating_sub(gen) > 60,
        };
        if stale {
            let measured = (self.latest.steps_per_sec > 1.0).then(|| {
                (self.latest.steps_per_sec, self.latest.snapshot.grid.len())
            });
            let mut est = [0.0f64; 3];
            for (k, opt) in PlumeLength::ALL.iter().enumerate() {
                est[k] = case::estimate_steady_seconds(
                    &self.params, *opt, &self.committed_wall, measured,
                );
            }
            self.plume_est = Some((est, self.frame_gen));
        }
        let est = self.plume_est.map(|(e, _)| e).unwrap_or_default();
        let mut clicked: Option<PlumeLength> = None;
        ui.horizontal_wrapped(|ui| {
            ui.label("Plume");
            for (k, opt) in PlumeLength::ALL.iter().enumerate() {
                let label = format!("{} · ~{}", opt.label(), fmt_duration(est[k]));
                let tip = match opt {
                    PlumeLength::Compact => "The historic domain: Mach disk plus half a shock cell.",
                    PlumeLength::Standard => "~20 exit radii of plume: Mach disk plus ~2 shock cells.",
                    PlumeLength::Long => "~40 exit radii of plume: 4-5 shock cells.",
                };
                let r = ui
                    .selectable_label(self.params.plume == *opt, label)
                    .on_hover_text(format!(
                        "{tip}\nEstimated time to visual steady state on this machine. \
                         Switching re-grades the grid and restarts the field."
                    ));
                if r.clicked() && self.params.plume != *opt {
                    clicked = Some(*opt);
                }
            }
        });
        ui.label(
            RichText::new(
                "graded grid: base cells across the geometry, 1.05 growth beyond \
                 — dt is unchanged, only the tail cells are added",
            )
            .weak()
            .small(),
        );
        if let Some(opt) = clicked {
            self.params.plume = opt;
            self.plume_est = None;
            let (lz, lr) = case::domain(&self.params);
            self.editor.set_domain(lz, lr);
            let setup = make_setup(&self.params, &self.committed_wall);
            let _ = self.tx.send(UiCommand::Rebuild(Box::new(setup)));
            self.relock_pending = true;
            self.canvas.request_fit();
        }
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

    fn engine_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Engine");
        ui.horizontal_wrapped(|ui| {
            let r = ui
                .selectable_label(self.preset.is_none(), "Custom")
                .on_hover_text(
                    "The demo case (docs §6). Moving any slider while a named \
                     engine is selected also lands here — a preset is applied \
                     whole or not at all.",
                );
            if r.clicked() && self.preset.is_some() {
                self.apply_preset(None);
            }
            for i in 0..PRESETS.len() {
                let p = &PRESETS[i];
                let label = if p.slow {
                    format!("{} (slow)", p.name)
                } else {
                    p.name.to_string()
                };
                let r = ui
                    .selectable_label(self.preset == Some(i), label)
                    .on_hover_text(preset_tooltip(p));
                if r.clicked() && self.preset != Some(i) {
                    self.apply_preset(Some(i));
                }
            }
        });
        // The gas model in effect. γ, T₀ and MW are labelled for what they
        // are: no manufacturer publishes combustion-gas properties per
        // engine, so these are propellant-class values, not measurements.
        let mw = R_UNIVERSAL_SI / self.params.r_specific_si;
        let head = match self.preset {
            Some(i) => format!("{} · {}", PRESETS[i].name, PRESETS[i].propellant),
            None => "custom gas".to_string(),
        };
        // What was PRODUCED, not what was asked for: on a rejected spec the
        // wall is the fallback cone and `wall_kind` says so.
        let source = match self.preset {
            Some(i) => format!(" — {}", PRESETS[i].bell_source),
            None => String::new(),
        };
        let contour_desc = match self.wall_kind {
            None => "hand-drawn wall".to_string(),
            Some(ContourKind::Conical) => "15° cone".to_string(),
            Some(ContourKind::ParabolicBell) => {
                format!("{:.0}% bell{source}", self.params.bell_percent * 100.0)
            }
            Some(ContourKind::MeasuredBell {
                theta_n_deg,
                theta_e_deg,
            }) => format!(
                "measured bell · θ_n {theta_n_deg:.1}° θ_e {theta_e_deg:.1}° · \
                 {:.0}% length{source}",
                self.params.bell_percent * 100.0
            ),
        };
        ui.label(
            RichText::new(format!(
                "{head} · r_t {:.0} mm · ε {:.1} · {contour_desc}",
                self.params.r_throat_m * 1e3,
                self.params.area_ratio
            ))
            .weak()
            .small(),
        );
        // Honesty flag: the wall on screen is not the wall that was asked for.
        // eprintln! into a windowed app's stdout is not a disclosure.
        if let Some(why) = &self.wall_fallback {
            ui.label(
                RichText::new("contour rejected — showing the 15° fallback cone")
                    .small()
                    .color(AMBER),
            )
            .on_hover_text(format!(
                "cfd_geom::generate_contour rejected this nozzle spec:\n{why}\n\n\
                 The wall being solved is the legacy 15° cone, not the requested \
                 contour. Every readout belongs to that cone.",
            ));
        }
        // Honesty flag: the digitised Rao table runs ε = 4..100 and rao_angles
        // CLAMPS at both ends rather than extrapolating — past either end the
        // bell is the end row's angles stretched to this exit radius. Merlin
        // Vac (ε = 165) lives above; the ε slider bottoms out at 2.0, below.
        // Measured-angle bells never touch the table, so they never flag.
        if self.wall_kind == Some(ContourKind::ParabolicBell)
            && !(4.0..=100.0).contains(&self.params.area_ratio)
        {
            let (end, side) = if self.params.area_ratio > 100.0 {
                (100.0, "ends at")
            } else {
                (4.0, "starts at")
            };
            ui.label(
                RichText::new(format!(
                    "bell angles clamped: Rao table {side} ε = {end:.0} (this nozzle is ε {:.1})",
                    self.params.area_ratio
                ))
                .small()
                .color(AMBER),
            )
            .on_hover_text(format!(
                "θ_n and θ_e come from the digitised Rao table, which covers \
                 ε = 4–100; outside it the angles clamp to the ε = {end:.0} row \
                 instead of extrapolating (the published data does not support \
                 extrapolation). The wall shown is that bell continued to this \
                 exit radius — plausible, but not a tabulated Rao contour.",
            ));
        }
        ui.label(
            RichText::new(format!(
                "γ {} · T₀ {:.0} K · MW {:.1} g/mol — propellant class, not measured",
                self.params.gamma, self.params.t0_k, mw
            ))
            .weak()
            .small(),
        )
        .on_hover_text(
            "No manufacturer publishes combustion-gas γ, chamber temperature \
             or molecular weight per engine. These are representative values \
             for the propellant class, and every readout inherits that \
             approximation.",
        );
        // Domain and cell counts from the frame's own grid — the truth about
        // what is being solved, including the graded plume tail.
        let g = &self.latest.snapshot.grid;
        ui.label(
            RichText::new(format!(
                "domain {:.0} × {:.0} r_t · {} × {} cells (graded)",
                g.lz(),
                g.lr(),
                g.nz,
                g.nr
            ))
            .weak()
            .small(),
        );
        self.plume_selector(ui);
    }

    fn operating_point(&mut self, ui: &mut egui::Ui) {
        ui.heading("Operating point");

        // ---- Altitude: never resets the field; the transient IS the demo.
        let sep_m = separation_altitude_m(&self.params);
        ui.label("Altitude");
        if altitude_slider(
            ui,
            &mut self.params.altitude_m,
            sep_m,
            &mut self.params.vacuum,
        ) {
            self.cmd(SolverCommand::SetAmbient(ambient_nd(&self.params)));
        }
        if self.params.vacuum {
            ui.label(
                RichText::new(format!(
                    "VACUUM · p∞ fixed at 3e-5 p₀ = {}",
                    fmt_pressure(VACUUM_P_FRAC * self.params.p0_pa)
                ))
                .small()
                .color(AMBER),
            )
            .on_hover_text(
                "Back pressure fixed at 30× the solver's positivity floor. \
                 True vacuum is unreachable for a finite-pressure Euler \
                 solver: with less margin the plume's expansion undershoot \
                 rides the floor at steady state and blanks every readout.",
            );
        } else {
            let (pa, ta) = atmosphere(self.params.altitude_m);
            let clamped = pa / self.params.p0_pa < VACUUM_P_FRAC;
            ui.label(
                RichText::new(format!(
                    "{:.1} km · p∞ {} · T∞ {:.0} K{}",
                    self.params.altitude_m / 1000.0,
                    fmt_pressure(pa),
                    ta,
                    if clamped {
                        " · clamped to 3e-5 p₀ (floor margin)"
                    } else {
                        ""
                    }
                ))
                .weak()
                .small(),
            );
        }
        ui.add_space(6.0);

        // ---- Area ratio (regenerates the parametric contour on release).
        // The range stretches to hold a selected preset (up to ε = 165);
        // egui would otherwise clamp the staged value back into 2..16.
        let ar_max = self.ui_area_ratio.max(16.0);
        let r = ui.add(
            egui::Slider::new(&mut self.ui_area_ratio, 2.0..=ar_max)
                .text("Area ratio ε")
                .fixed_decimals(1),
        );
        if r.drag_stopped() || (r.changed() && !r.dragged()) {
            self.regenerate_contour();
        }

        // ---- Chamber pressure (a field-discarding control).
        let p0_max = self.ui_p0_mpa.max(10.0);
        let r = ui.add(
            egui::Slider::new(&mut self.ui_p0_mpa, 0.5..=p0_max)
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
            // The product rule, quarantine form (docs §13): nothing displays
            // while any floor activation is recent — the field near the floor
            // is actively being invented. Startup activations older than the
            // quarantine display with a permanent amber disclosure instead:
            // the high-pressure presets' cold-start front brushes the 1e-6
            // floor by design (§5), stops within ~800 steps, and the invented
            // mass has long left the domain.
            let quiet = info.step.saturating_sub(self.floor_last_step);
            if quiet < FLOOR_QUARANTINE_STEPS {
                ui.colored_label(RED, RichText::new("SOLUTION INVALID").strong());
                ui.label(format!(
                    "Positivity floor activated {} time(s), most recently {} \
                     step(s) ago. No quantitative output will be shown until \
                     the floor has been quiet for {} steps.",
                    info.floor_activations, quiet, FLOOR_QUARANTINE_STEPS
                ));
                return;
            }
            ui.colored_label(
                AMBER,
                RichText::new(format!(
                    "floor tripped {}× during startup · none in the last {} steps",
                    info.floor_activations, quiet
                ))
                .small(),
            )
            .on_hover_text(
                "The cold-start front of a high-pressure engine transiently \
                 drops below the 1e-6 p₀ pressure floor (§5) and those cells \
                 were clamped. The startup transient was therefore not \
                 conservative; the steady state shown now never touches the \
                 floor and its integrals are unaffected.",
            );
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
                    // Honesty rules (docs §13): absolute thrust is never
                    // displayed for a named real engine — this model has no
                    // boundary layer, chemistry or divergence correction to
                    // defend such a number. C_f below is the honest ratio.
                    if self.preset.is_some() {
                        ui.monospace("—");
                        ui.label(
                            RichText::new(format!(
                                "{n_throat:.0} cells / r_t · withheld for a named engine — see C_f"
                            ))
                            .small()
                            .color(if underresolved {
                                AMBER
                            } else {
                                ui.visuals().weak_text_color()
                            }),
                        );
                    } else {
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
                    }
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
            let prev_step = self.latest.info.step;
            self.latest = self.out.read().clone();
            self.frame_gen += 1;
            // The worker pauses itself on a solver error; mirror that here so
            // the transport bar tells the truth.
            if self.latest.error.is_some() && self.latest.paused {
                self.paused = true;
            }
            // Frames from a rebuilt solver (step count restarted): arm the
            // re-lock, then fire it once the startup transient has developed
            // a usable range — the init field is near-quiescent and locking
            // to it saturates the display.
            if self.relock_pending
                && (self.latest.info.step < prev_step || self.latest.info.step <= 2)
            {
                self.relock_pending = false;
                self.relock_armed = true;
            }
            if self.relock_armed && self.latest.info.step >= RELOCK_STEP {
                for k in FieldKind::ALL {
                    self.locked[k as usize] = lock_range(k, self.latest.snapshot.range(k));
                }
                self.relock_armed = false;
            }
            // Floor-activation quarantine bookkeeping.
            let info = &self.latest.info;
            if info.floor_activations < self.floors_seen {
                // Rebuilt or reset solver: fresh counter.
                self.floors_seen = 0;
                self.floor_last_step = 0;
            }
            if info.floor_activations > self.floors_seen
                || (info.step < prev_step && self.floors_seen > 0)
            {
                // New activations — or an ambient change restarted the step
                // count with a nonzero counter (a new transient): re-arm.
                self.floors_seen = info.floor_activations;
                self.floor_last_step = info.step;
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
                    self.engine_section(ui);
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

fn fmt_duration(secs: f64) -> String {
    if !secs.is_finite() || secs <= 0.0 {
        "—".into()
    } else if secs < 90.0 {
        format!("{secs:.0} s")
    } else {
        format!("{:.1} min", secs / 60.0)
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

fn preset_tooltip(p: &case::EnginePreset) -> String {
    let shape = match p.contour_kind {
        ContourKind::Conical => "15° cone".to_string(),
        ContourKind::ParabolicBell => format!("{:.0}% parabolic bell", p.bell_percent * 100.0),
        ContourKind::MeasuredBell {
            theta_n_deg,
            theta_e_deg,
        } => format!(
            "measured bell, θ_n {theta_n_deg:.1}° / θ_e {theta_e_deg:.1}° on a \
             {:.3} R_t throat arc, {:.0}% length",
            p.throat_arc_down,
            p.bell_percent * 100.0
        ),
    };
    let mut s = format!(
        "{} · ε {} · p₀ {:.0} bar · r_t {:.0} mm\n{shape} — {}\n≈{:.1}× Merlin 1D run time",
        p.propellant,
        p.area_ratio,
        p.p0_pa / 1e5,
        p.r_throat_m * 1e3,
        p.bell_source,
        p.relative_cost()
    );
    // The Rao θ_n/θ_e table is digitised at γ ≈ 1.23–1.25
    // (docs/physics-reference.md §6, §10); an engine running well outside that
    // band inherits the mismatch in its wall shape — a small one: Rao (1958)
    // has γ barely moving the contour at fixed length and area ratio, worth
    // well under a degree of exit angle. Only table bells can inherit it at
    // all; a measured-angle contour never consults the table.
    if p.contour_kind == ContourKind::ParabolicBell && (p.gamma < 1.23 || p.gamma > 1.25) {
        s.push_str(&format!(
            "\nBell angles from the Rao table, digitised at γ 1.23–1.25; this \
             engine runs γ {} — the wall shape inherits that mismatch (sub-degree).",
            p.gamma
        ));
    }
    if !p.note.is_empty() {
        s.push('\n');
        s.push_str(p.note);
    }
    s
}

/// Custom altitude slider: the track is shaded below the separation crossing
/// and carries a labeled tick there, recomputed as the area ratio moves, so
/// the user sees where the trustworthy range ends BEFORE dragging. The track
/// caps at 58 km (docs §5); past its right end sits the labelled vacuum stop
/// with a fixed back pressure.
fn altitude_slider(
    ui: &mut egui::Ui,
    alt_m: &mut f64,
    sep_m: Option<f64>,
    vacuum: &mut bool,
) -> bool {
    let width = ui.available_width().min(300.0);
    let (rect, response) = ui.allocate_exact_size(Vec2::new(width, 48.0), Sense::click_and_drag());
    let track_y = rect.top() + 26.0;
    // Rightmost stop is the vacuum zone, past the 58 km end of the track.
    const VAC_W: f32 = 30.0;
    let (x0, x1) = (rect.left() + 8.0, rect.right() - 8.0 - VAC_W - 8.0);
    let x_of = |m: f64| x0 + ((m / ALT_MAX_M) as f32) * (x1 - x0);

    let mut changed = false;
    if response.dragged() || response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            if pos.x > x1 + 4.0 {
                if !*vacuum {
                    *vacuum = true;
                    changed = true;
                }
            } else {
                let m = (((pos.x - x0) / (x1 - x0)) as f64 * ALT_MAX_M).clamp(0.0, ALT_MAX_M);
                if *vacuum || (m - *alt_m).abs() > 1.0 {
                    *alt_m = m;
                    *vacuum = false;
                    changed = true;
                }
            }
        }
    }

    let painter = ui.painter_at(rect);
    painter.line_segment(
        [Pos2::new(x0, track_y), Pos2::new(x1, track_y)],
        Stroke::new(4.0, ui.visuals().widgets.inactive.bg_fill),
    );
    // The vacuum stop.
    let vac_rect = Rect::from_min_max(
        Pos2::new(x1 + 8.0, track_y - 9.0),
        Pos2::new(x1 + 8.0 + VAC_W, track_y + 9.0),
    );
    let (vac_fill, vac_stroke) = if *vacuum {
        (AMBER.gamma_multiply(0.25), AMBER)
    } else {
        (
            ui.visuals().widgets.inactive.bg_fill.gamma_multiply(0.5),
            ui.visuals().weak_text_color(),
        )
    };
    painter.rect_filled(vac_rect, 3.0, vac_fill);
    painter.rect_stroke(
        vac_rect,
        3.0,
        Stroke::new(1.0, vac_stroke),
        StrokeKind::Outside,
    );
    painter.text(
        vac_rect.center(),
        Align2::CENTER_CENTER,
        "vac",
        FontId::proportional(10.0),
        if *vacuum {
            AMBER
        } else {
            ui.visuals().weak_text_color()
        },
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
    // 10 km ticks, plus the 58 km cap.
    for km in [0, 10, 20, 30, 40, 50, 58] {
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
    // Handle: parked in the vacuum stop when vacuum mode is on.
    if !*vacuum {
        let hx = x_of(*alt_m);
        painter.circle(
            Pos2::new(hx, track_y),
            7.0,
            ui.visuals().widgets.active.bg_fill,
            Stroke::new(1.5, ui.visuals().widgets.active.fg_stroke.color),
        );
    }
    changed
}
