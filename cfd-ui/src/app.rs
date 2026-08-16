//! The application: layout, controls, keybindings and the honesty surface.
//! docs/sessions/D-ui.md §3–§4 are the authority on what may and may not be
//! shown; docs/physics-reference.md §13 on the wording.

use std::sync::mpsc::Sender;
use std::sync::Arc;

use eframe::egui::{
    self, Align2, Color32, FontId, Key, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, Vec2,
};

use cfd_contract::{FieldKind, SolverCommand};

use crate::canvas::{colorbar, fmt_value, range_drag, Canvas, RangeEdit, RangeInvalid, BG};
use crate::case::{
    self, ambient_nd, atmosphere, conical_contour, ideal_cf, make_setup, rasterize_wall,
    separation_altitude_m, separation_threshold, CaseParams, DomainPreset, ALT_MAX_M, PRESETS,
    R_UNIVERSAL_SI, VACUUM_P_FRAC,
};
use crate::colormap::{Preset, PresetKind};
use crate::editor::{EditorBackend, StubEditor};
use crate::worker::{UiCommand, UiFrame};

/// Per-field state attached to `CfdApp::locked`.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct RangeState {
    /// The range was typed or dragged rather than set by `lock_range`. Manual
    /// ranges survive a re-lock; auto ones are overwritten by it.
    manual: bool,
    /// A manual range survived a re-lock, so the scales it was chosen against
    /// may no longer exist. Cleared by Refit or by any further edit.
    stale: bool,
    /// Guardrails suspended for this field: the Schlieren pin and the signed
    /// symmetry stop being enforced, and the caveat is shown instead.
    free: bool,
    /// Ends whose last edit was rejected.
    invalid: RangeInvalid,
}

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
    /// Per-field colormap choice. Decoupled from `FieldKind` — the style guide
    /// §2 table is the default, not a fixed assignment.
    cmap: [Preset; 8],
    /// Per-field bookkeeping for `locked`, parallel to it by field index.
    range_state: [RangeState; 8],
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
    // Domain-size staging (work order: configurable-domain): the four sidebar
    // fields, committed on release / focus loss like the sliders. `params`
    // holds the values the solver is actually running.
    ui_lz_rt: f64,
    ui_lr_rt: f64,
    ui_cells_per_rt: f64,
    ui_dz_over_dr: f64,
    /// Cached live cost readout: (staged field values, ui tick it was
    /// computed at, estimate). Each estimate re-runs grading + wall
    /// rasterization, so it refreshes on a bounded cadence, not per repaint.
    cost_cache: Option<([f64; 4], u64, case::CostEstimate)>,
    /// Repaint counter for the cost-readout cadence (frame_gen freezes when
    /// the worker pauses; this does not).
    ui_tick: u64,
    /// The staged domain settings exceed the machine's memory: the rebuild is
    /// blocked until the user explicitly confirms or reverts.
    mem_confirm: bool,
    /// Engine-preset tooltips, built once: `relative_cost` re-runs the
    /// grading for its cost model, far too slow for a per-repaint call.
    preset_tips: Vec<String>,
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
        let mut cmap = [Preset::Viridis; 8];
        for k in FieldKind::ALL {
            locked[k as usize] = lock_range(k, initial.snapshot.range(k));
            cmap[k as usize] = Preset::default_for(k);
        }
        let mut editor = StubEditor::new(wall.clone());
        editor.set_domain(params.lz_rt, params.lr_rt);
        CfdApp {
            out,
            tx,
            latest: initial,
            frame_gen: 0,
            ui_area_ratio: params.area_ratio,
            ui_p0_mpa: params.p0_pa / 1e6,
            ui_lz_rt: params.lz_rt,
            ui_lr_rt: params.lr_rt,
            ui_cells_per_rt: params.cells_per_rt,
            ui_dz_over_dr: params.dz_over_dr,
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
            cmap,
            range_state: [RangeState::default(); 8],
            smooth: false,
            canvas: Canvas::new(),
            editor,
            editor_on: false,
            committed_wall: wall,
            geometry_custom: false,
            space_panned: false,
            hover_text: String::new(),
            cost_cache: None,
            ui_tick: 0,
            mem_confirm: false,
            preset_tips: PRESETS.iter().map(preset_tooltip).collect(),
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
        let wall = conical_contour(self.params.area_ratio);
        self.editor.set_points(wall.clone());
        self.committed_wall = wall;
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
        self.sync_domain_fields();
        let wall = conical_contour(self.params.area_ratio);
        self.editor.set_domain(self.params.lz_rt, self.params.lr_rt);
        self.editor.set_points(wall.clone());
        self.committed_wall = wall;
        self.geometry_custom = false;
        let setup = make_setup(&self.params, &self.committed_wall);
        let _ = self.tx.send(UiCommand::Rebuild(Box::new(setup)));
        self.relock_pending = true;
        self.canvas.request_fit(); // the domain just changed size
    }

    /// Mirror the committed params into the staged sidebar fields.
    fn sync_domain_fields(&mut self) {
        self.ui_lz_rt = self.params.lz_rt;
        self.ui_lr_rt = self.params.lr_rt;
        self.ui_cells_per_rt = self.params.cells_per_rt;
        self.ui_dz_over_dr = self.params.dz_over_dr;
        self.mem_confirm = false;
    }

    /// The staged sidebar values as a candidate case, sanitized (clamped
    /// ranges; non-finite entry falls back to the committed value) so the
    /// cost estimator and the solver can never see NaN or a zero-size grid.
    fn staged_params(&mut self) -> CaseParams {
        let mut cand = CaseParams {
            lz_rt: self.ui_lz_rt,
            lr_rt: self.ui_lr_rt,
            cells_per_rt: self.ui_cells_per_rt,
            dz_over_dr: self.ui_dz_over_dr,
            ..self.params
        };
        case::sanitize_domain(&mut cand, &self.params);
        self.ui_lz_rt = cand.lz_rt;
        self.ui_lr_rt = cand.lr_rt;
        self.ui_cells_per_rt = cand.cells_per_rt;
        self.ui_dz_over_dr = cand.dz_over_dr;
        cand
    }

    /// Commit the staged domain settings: rebuild and restart on the
    /// re-graded grid, exactly as the old tier buttons did. Blocked behind an
    /// explicit confirmation when the estimate exceeds the machine's memory
    /// (`force` is that confirmation).
    fn commit_domain(&mut self, force: bool) {
        let cand = self.staged_params();
        if cand == self.params {
            self.mem_confirm = false;
            return;
        }
        let est = case::estimate_cost(&cand, &self.committed_wall, self.throughput());
        if !force && est.bytes > case::memory_budget_bytes() {
            self.mem_confirm = true;
            return;
        }
        self.mem_confirm = false;
        self.params = cand;
        self.cost_cache = None;
        self.editor.set_domain(self.params.lz_rt, self.params.lr_rt);
        let setup = make_setup(&self.params, &self.committed_wall);
        let _ = self.tx.send(UiCommand::Rebuild(Box::new(setup)));
        self.relock_pending = true;
        self.canvas.request_fit();
    }

    /// This machine's measured solver throughput in cells/s, if the worker's
    /// calibration run has reported yet.
    fn throughput(&self) -> Option<f64> {
        (self.latest.cells_per_sec > 0.0).then_some(self.latest.cells_per_sec)
    }

    /// Domain size and mesh resolution (work order: configurable-domain):
    /// a live cost readout above four editable fields, plus three shortcut
    /// buttons that just fill the fields. Committing a change re-grades the
    /// grid, rebuilds the solver and restarts the field.
    fn domain_section(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Domain size").strong());

        // ---- live cost readout, from the STAGED values, before any solve.
        // Bounded cadence: each estimate re-runs grading + rasterization.
        let staged = self.staged_params();
        let key = [staged.lz_rt, staged.lr_rt, staged.cells_per_rt, staged.dz_over_dr];
        let stale = match &self.cost_cache {
            None => true,
            Some((k, tick, _)) => {
                let age = self.ui_tick.saturating_sub(*tick);
                (*k != key && age >= 10) || age > 120
            }
        };
        if stale {
            let est = case::estimate_cost(&staged, &self.committed_wall, self.throughput());
            self.cost_cache = Some((key, self.ui_tick, est));
        }
        let est = self.cost_cache.as_ref().unwrap().2;
        ui.label(
            RichText::new(format!(
                "{} cells ({} × {} graded) · ~{} steps to steady",
                group_thousands(est.cells as u64),
                est.nz,
                est.nr,
                group_thousands(est.steps.round() as u64),
            ))
            .small(),
        );
        let time_txt = if est.seconds.is_finite() {
            format!("~{} wall clock", fmt_duration(est.seconds))
        } else {
            "wall clock: measuring this machine…".to_string()
        };
        ui.label(RichText::new(format!("{time_txt} · ~{} memory", fmt_bytes(est.bytes))).small())
            .on_hover_text(
                "Estimated from the §9 visual-steady step count scaled to this \
                 domain and dt, and from a solver throughput measured on THIS \
                 machine at the last rebuild. An estimate, not a promise.",
            );
        if est.seconds.is_finite() && est.seconds > 1800.0 {
            ui.colored_label(
                AMBER,
                RichText::new(format!(
                    "~{} — over the ~30 min budget for a normal run",
                    fmt_duration(est.seconds)
                ))
                .small(),
            );
        }

        // ---- the four fields. Committed on release / focus loss, like the
        // sliders; DragValue clamps typed input into the same ranges that
        // `sanitize_domain` enforces.
        let mut commit = false;
        ui.horizontal(|ui| {
            ui.label("Resolution");
            let r = ui.add(
                egui::DragValue::new(&mut self.ui_cells_per_rt)
                    .range(case::CELLS_PER_RT_RANGE.0..=case::CELLS_PER_RT_RANGE.1)
                    .speed(0.5)
                    .fixed_decimals(0)
                    .suffix(" cells/r_t"),
            )
            .on_hover_text(
                "Base radial cells across the throat radius. The mass-flow \
                 badge goes amber below 20 (§8).",
            );
            commit |= r.drag_stopped() || r.lost_focus();
            ui.label("· aspect dz/dr");
            let r = ui.add(
                egui::DragValue::new(&mut self.ui_dz_over_dr)
                    .range(case::DZ_OVER_DR_RANGE.0..=case::DZ_OVER_DR_RANGE.1)
                    .speed(0.05)
                    .fixed_decimals(1),
            )
            .on_hover_text(
                "Axial cells are this many times wider than radial ones. \
                 dt is set by the radial spacing, so wide-in-z is nearly free \
                 (§8); narrow it toward 2.0 to resolve the throat arc better.",
            );
            commit |= r.drag_stopped() || r.lost_focus();
        });
        ui.horizontal(|ui| {
            ui.label("Domain");
            let r = ui.add(
                egui::DragValue::new(&mut self.ui_lz_rt)
                    .range(case::LZ_RANGE.0..=case::LZ_RANGE.1)
                    .speed(1.0)
                    .fixed_decimals(0),
            )
            .on_hover_text("Domain length in throat radii.");
            commit |= r.drag_stopped() || r.lost_focus();
            ui.label("×");
            let r = ui.add(
                egui::DragValue::new(&mut self.ui_lr_rt)
                    .range(case::LR_RANGE.0..=case::LR_RANGE.1)
                    .speed(0.5)
                    .fixed_decimals(0),
            )
            .on_hover_text("Domain radius in throat radii.");
            commit |= r.drag_stopped() || r.lost_focus();
            ui.label(RichText::new("r_t (length × radius)").weak().small());
        });

        // ---- shortcuts: fill the fields and commit. Not tiers — the fields
        // stay editable afterwards.
        ui.horizontal_wrapped(|ui| {
            for opt in DomainPreset::ALL {
                let (lz, lr, n, asp) = opt.values();
                let selected = self.params.lz_rt == lz
                    && self.params.lr_rt == lr
                    && self.params.cells_per_rt == n
                    && self.params.dz_over_dr == asp;
                let tip = match opt {
                    DomainPreset::Preview => {
                        "The historic compact interactive domain — returns in \
                         seconds; Mach disk plus half a shock cell."
                    }
                    DomainPreset::Standard => "Half the Large domain in both directions.",
                    DomainPreset::Large => "The full domain (the default).",
                };
                let r = ui
                    .selectable_label(selected, opt.label())
                    .on_hover_text(format!(
                        "{tip}\n{lz:.0} × {lr:.0} r_t at {n:.0} cells/r_t — fills the \
                         fields above; they stay editable."
                    ));
                if r.clicked() && !selected {
                    self.ui_lz_rt = lz;
                    self.ui_lr_rt = lr;
                    self.ui_cells_per_rt = n;
                    self.ui_dz_over_dr = asp;
                    commit = true;
                }
            }
        });
        ui.label(
            RichText::new(
                "graded grid: base cells across the geometry and plume core, \
                 1.05 growth beyond — dt is unchanged, only far-field cells \
                 are added",
            )
            .weak()
            .small(),
        );

        if commit {
            self.commit_domain(false);
        }

        // ---- blocking confirmation: the staged settings would exhaust
        // memory. Nothing rebuilds until the user decides.
        if self.mem_confirm {
            warning_box(
                ui,
                RED,
                "TOO BIG FOR THIS MACHINE",
                &format!(
                    "These settings need ~{} but only ~{} of memory is \
                     available. The app may crash or thrash if you proceed.",
                    fmt_bytes(est.bytes),
                    fmt_bytes(case::memory_budget_bytes())
                ),
            );
            ui.horizontal(|ui| {
                if ui.button("Run anyway").clicked() {
                    self.commit_domain(true);
                }
                if ui.button("Revert").clicked() {
                    self.sync_domain_fields();
                }
            });
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

    /// The single writer for `locked`, shared by every range editor — the
    /// in-place colorbar labels and the explicit min/max row. Two views, one
    /// value: they cannot drift because neither of them owns it.
    ///
    /// A rejected edit reverts and highlights rather than clamping. Silently
    /// clamping would misreport what the player asked for, and on a signed
    /// field it is not even well defined — the symmetry rule would undo the
    /// clamp on the next edit anyway.
    fn apply_range_edit(&mut self, field: FieldKind, edit: RangeEdit) {
        let i = field as usize;
        let (min_end, v) = match edit {
            RangeEdit::None => return,
            RangeEdit::Min(v) => (true, v),
            RangeEdit::Max(v) => (false, v),
        };
        let free = self.range_state[i].free;
        match edited_range(field, free, self.locked[i], min_end, v) {
            Some(r) => {
                self.locked[i] = r;
                self.range_state[i].manual = true;
                self.range_state[i].stale = false;
                self.range_state[i].invalid = RangeInvalid::default();
            }
            None => self.range_state[i].invalid.set(min_end, true),
        }
    }

    /// Refit hands the range back to `lock_range`, which also clears `manual` —
    /// so a later re-lock is free to overwrite it again.
    fn refit_range(&mut self, field: FieldKind) {
        let i = field as usize;
        let r = self.latest.snapshot.range(field);
        self.locked[i] = if self.range_state[i].free {
            // Guardrails suspended: fit the data as it actually is.
            if r.1 > r.0 {
                r
            } else {
                (r.0, r.0 + 1.0)
            }
        } else {
            lock_range(field, r)
        };
        self.range_state[i] = RangeState {
            free: self.range_state[i].free,
            ..RangeState::default()
        };
    }

    /// Turning guardrails back on re-imposes them immediately. Leaving an
    /// asymmetric range in place under a rule that claims symmetry would make
    /// the toggle a lie about what is being displayed.
    fn set_free_range(&mut self, field: FieldKind, free: bool) {
        let i = field as usize;
        self.range_state[i].free = free;
        self.range_state[i].invalid = RangeInvalid::default();
        if !free {
            self.locked[i] = lock_range(field, self.locked[i]);
        }
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

        let field = self.field;
        let i = field as usize;
        let mut edit = RangeEdit::None;
        ui.horizontal(|ui| {
            edit = colorbar(
                ui,
                self.cmap[i],
                self.locked[i],
                120.0,
                self.range_state[i].invalid,
            );
            ui.vertical(|ui| {
                ui.label(RichText::new(field.label()).strong());
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
                    self.refit_range(field);
                }
                if self.range_state[i].stale {
                    ui.label(
                        RichText::new("scales changed — range may be stale")
                            .weak()
                            .small()
                            .color(AMBER),
                    )
                    .on_hover_text(
                        "This range was typed, so the re-lock after the rebuild \
                         left it alone. A p₀ change can move the reference \
                         scales by 35×, which renders a stale range as a solid \
                         block of one colour. Refit to clear.",
                    );
                }
                ui.checkbox(&mut self.smooth, "Smooth (L)")
                    .on_hover_text("LINEAR filtering; default NEAREST shows cells");
            });
        });
        self.apply_range_edit(field, edit);

        // ---- colormap picker. Every preset is offered for every field: the
        // guidance is carried by the warning below, not by filtering the list.
        ui.horizontal(|ui| {
            ui.label("Colormap");
            egui::ComboBox::from_id_salt("cmap_pick")
                .selected_text(self.cmap[i].name())
                .show_ui(ui, |ui| {
                    for p in Preset::ALL {
                        ui.selectable_value(&mut self.cmap[i], p, p.name());
                    }
                });
        });
        if self.cmap[i].kind() == PresetKind::Diverging && !field.is_signed() {
            ui.label(
                RichText::new(
                    "diverging map on an unsigned field — the light band lands \
                     at an arbitrary value and reads as a feature that is not there",
                )
                .weak()
                .small(),
            )
            .on_hover_text(
                "Legitimate if you mean it: a diverging map centred on Mach 1.0 \
                 is a real transonic view. It is only misleading when the centre \
                 is wherever the range happens to put it.",
            );
        }

        // ---- explicit range row, the second view of `locked`.
        let mut row_edit = RangeEdit::None;
        ui.horizontal(|ui| {
            let (lo, hi) = self.locked[i];
            let span = hi - lo;
            let invalid = self.range_state[i].invalid;
            ui.label("min");
            if let Some(v) = range_drag(ui, lo, span, invalid.min, None) {
                row_edit = RangeEdit::Min(v);
            }
            ui.label("max");
            if let Some(v) = range_drag(ui, hi, span, invalid.max, None) {
                row_edit = RangeEdit::Max(v);
            }
        });
        self.apply_range_edit(field, row_edit);

        // Only Schlieren and the signed velocities carry a guardrail; on any
        // other field the toggle would be a control that does nothing.
        if field == FieldKind::Schlieren || field.is_signed() {
            let mut free = self.range_state[i].free;
            if ui
                .checkbox(&mut free, "Free range")
                .on_hover_text(
                    "Suspend the style guide's range rule for this field \
                     (docs/colormap-style-guide.md §8).",
                )
                .changed()
            {
                self.set_free_range(field, free);
            }
            if self.range_state[i].free {
                let caveat = if field.is_signed() {
                    "asymmetric — zero is off the colormap's light point"
                } else {
                    "unpinned — Schlieren is renormalised to [0, 1] every frame"
                };
                ui.label(RichText::new(caveat).weak().small().color(AMBER));
            }
        }
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
                    .on_hover_text(&self.preset_tips[i]);
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
        ui.label(
            RichText::new(format!(
                "{head} · r_t {:.0} mm · ε {:.1}",
                self.params.r_throat_m * 1e3,
                self.params.area_ratio
            ))
            .weak()
            .small(),
        );
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
        self.domain_section(ui);
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
        // Resolution-dependent badges (work order: configurable-domain).
        // Mass-flow quantization is ±1/N_throat of the throat area (§8 table:
        // ±5% at 20 cells/r_t, ±2.5% at 40). The staircase-wall biases are
        // FIRST order in cell size, so the T8-measured figures at the
        // N_throat = 20 reference (≈13% thrust, ≈19% exit Mach) scale as
        // 20/N_throat — quoted numbers would lie at any other resolution.
        let scale_ok = n_throat.is_finite() && n_throat >= 1.0;
        let mdot_band = if scale_ok {
            format!("±{:.1}%", 100.0 / n_throat)
        } else {
            "±?%".into()
        };
        let thrust_bias = 13.0 * 20.0 / n_throat;
        let mach_bias = 19.0 * 20.0 / n_throat;

        // Greyed-out whenever unsettled: a report from an unconverged field
        // is not a number.
        ui.add_enabled_ui(settled, |ui| {
            egui::Grid::new("report")
                .num_columns(3)
                .spacing([10.0, 3.0])
                .show(ui, |ui| {
                    ui.label("mass flow");
                    ui.monospace(fmt_or_dash(rep.mass_flow_kg_s, "kg/s"));
                    ui.label(RichText::new(&mdot_band).small().color(if underresolved {
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
                        // The bias reference is measured (T8, ladder.rs): the
                        // staircase wall's entropy layer costs ~13% of thrust
                        // at N_throat = 20, first order in cell size.
                        ui.label(
                            RichText::new(if scale_ok {
                                format!(
                                    "{n_throat:.0} cells / r_t · ≈{thrust_bias:.0}% low \
                                     (staircase wall)"
                                )
                            } else {
                                "resolution unknown".into()
                            })
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
                    // ~19% below the 1-D ideal at N_throat = 20, first order
                    // in cell size; the core flow is within ~7%.
                    ui.label(
                        RichText::new(if scale_ok {
                            format!(
                                "area-avg · ≈{mach_bias:.0}% low at this res (staircase \
                                 wall) · 1-D ideal {:.2}",
                                rep.ideal_exit_mach
                            )
                        } else {
                            format!("area-avg · 1-D ideal {:.2}", rep.ideal_exit_mach)
                        })
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
        self.ui_tick += 1;
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
                // A typed range is left alone, and flagged. A p₀ change can
                // move the reference scales by 35×, so a range chosen against
                // the old ones can render as a solid block of a single colour.
                // Discarding what the player typed loses their work; showing
                // the solid block with no explanation loses their trust.
                // Flagging is the only option that loses neither.
                for k in FieldKind::ALL {
                    let i = k as usize;
                    let fresh = self.latest.snapshot.range(k);
                    relock_field(k, &mut self.range_state[i], &mut self.locked[i], fresh);
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
                    self.cmap[self.field as usize],
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

/// The range-edit rule as a pure function: `None` rejects the edit and leaves
/// the committed range alone. `min_end` says which end the player touched.
///
/// Separate from `CfdApp` so the rules are testable without an event loop.
fn edited_range(
    field: FieldKind,
    free: bool,
    cur: (f32, f32),
    min_end: bool,
    v: f32,
) -> Option<(f32, f32)> {
    // Non-finite reverts to the committed value, the same way
    // `case::sanitize_domain` treats the domain fields. Schlieren arrives
    // renormalised every frame, so with guardrails on there is nothing to edit.
    if !v.is_finite() || (field == FieldKind::Schlieren && !free) {
        return None;
    }
    let cand = if field.is_signed() && !free {
        // Symmetric about zero: moving one end moves the other to match, so
        // zero stays on the diverging map's light point.
        let m = v.abs();
        (-m, m)
    } else if min_end {
        (v, cur.1)
    } else {
        (cur.0, v)
    };
    // Rejected, not clamped: an inverted range is a question, not a typo to
    // guess at.
    (cand.0 < cand.1).then_some(cand)
}

/// The re-lock rule for one field. Auto ranges are overwritten; a manual one
/// survives and is flagged instead.
fn relock_field(kind: FieldKind, st: &mut RangeState, locked: &mut (f32, f32), fresh: (f32, f32)) {
    if st.manual {
        st.stale = true;
    } else {
        *locked = lock_range(kind, fresh);
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

fn fmt_bytes(b: u64) -> String {
    if b >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", b as f64 / (1024.0 * 1024.0 * 1024.0))
    } else {
        format!("{:.0} MB", (b as f64 / (1024.0 * 1024.0)).max(1.0))
    }
}

fn group_thousands(v: u64) -> String {
    let s = v.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
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
    let mut s = format!(
        "{} · ε {} · p₀ {:.0} bar · r_t {:.0} mm\n≈{:.1}× Merlin 1D run time",
        p.propellant,
        p.area_ratio,
        p.p0_pa / 1e5,
        p.r_throat_m * 1e3,
        p.relative_cost()
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    const SIGNED: FieldKind = FieldKind::VelocityZ;
    const UNSIGNED: FieldKind = FieldKind::Mach;

    #[test]
    fn lock_range_applies_the_style_guide_rules() {
        // Schlieren is pinned regardless of what the frame reports.
        assert_eq!(lock_range(FieldKind::Schlieren, (-3.0, 900.0)), (0.0, 1.0));
        // Signed fields come back symmetric about zero.
        assert_eq!(lock_range(SIGNED, (-120.0, 40.0)), (-120.0, 120.0));
        assert_eq!(lock_range(SIGNED, (10.0, 90.0)), (-90.0, 90.0));
        // Unsigned fields pass through, with a degenerate range widened.
        assert_eq!(lock_range(UNSIGNED, (0.0, 4.0)), (0.0, 4.0));
        assert_eq!(lock_range(UNSIGNED, (2.0, 2.0)), (2.0, 3.0));
    }

    #[test]
    fn guardrails_keep_a_signed_field_symmetric_under_a_one_ended_edit() {
        // Editing only the max moves the min to match.
        let r = edited_range(SIGNED, false, (-100.0, 100.0), false, 250.0).unwrap();
        assert_eq!(r, (-250.0, 250.0));
        // ... and editing only the min does the same, sign-independently.
        let r = edited_range(SIGNED, false, (-100.0, 100.0), true, -30.0).unwrap();
        assert_eq!(r, (-30.0, 30.0));
        let r = edited_range(SIGNED, false, (-100.0, 100.0), true, 30.0).unwrap();
        assert_eq!(r, (-30.0, 30.0));
    }

    #[test]
    fn free_range_lets_a_signed_field_go_asymmetric() {
        let r = edited_range(SIGNED, true, (-100.0, 100.0), false, 250.0).unwrap();
        assert_eq!(r, (-100.0, 250.0), "the untouched end must not move");
        let r = edited_range(SIGNED, true, (-100.0, 100.0), true, -5.0).unwrap();
        assert_eq!(r, (-5.0, 100.0));
    }

    #[test]
    fn schlieren_is_pinned_until_free_range_is_on() {
        assert_eq!(
            edited_range(FieldKind::Schlieren, false, (0.0, 1.0), false, 4.0),
            None
        );
        assert_eq!(
            edited_range(FieldKind::Schlieren, true, (0.0, 1.0), false, 4.0),
            Some((0.0, 4.0))
        );
    }

    #[test]
    fn non_finite_and_inverted_edits_are_rejected_not_clamped() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(edited_range(UNSIGNED, false, (0.0, 4.0), true, bad), None);
            assert_eq!(edited_range(UNSIGNED, false, (0.0, 4.0), false, bad), None);
        }
        // min pushed past max, and max pulled below min: both rejected whole,
        // never silently clamped to the other end.
        assert_eq!(edited_range(UNSIGNED, false, (0.0, 4.0), true, 9.0), None);
        assert_eq!(edited_range(UNSIGNED, false, (0.0, 4.0), false, -9.0), None);
        // An empty range is rejected at the boundary too.
        assert_eq!(edited_range(UNSIGNED, false, (0.0, 4.0), true, 4.0), None);
        // A symmetric edit of zero width is empty, so it is rejected as well.
        assert_eq!(edited_range(SIGNED, false, (-1.0, 1.0), false, 0.0), None);
    }

    #[test]
    fn relock_overwrites_auto_ranges_and_spares_manual_ones() {
        // Auto: overwritten by the fresh scales, and stays unflagged.
        let mut st = RangeState::default();
        let mut locked = (0.0, 4.0);
        relock_field(UNSIGNED, &mut st, &mut locked, (0.0, 140.0));
        assert_eq!(locked, (0.0, 140.0));
        assert!(!st.stale, "an auto range has nothing to go stale");
        assert!(!st.manual);

        // Manual: survives the relock untouched, and is flagged instead.
        let mut st = RangeState {
            manual: true,
            ..RangeState::default()
        };
        let mut locked = (0.0, 4.0);
        relock_field(UNSIGNED, &mut st, &mut locked, (0.0, 140.0));
        assert_eq!(locked, (0.0, 4.0), "a typed range must not be discarded");
        assert!(st.stale, "a surviving manual range must be flagged");
    }

    #[test]
    fn editing_clears_the_stale_flag() {
        // The flag means "typed against scales that have since moved". Any
        // fresh edit re-answers that, so it must clear.
        let mut st = RangeState {
            manual: true,
            stale: true,
            ..RangeState::default()
        };
        let mut locked = (0.0, 4.0);
        if let Some(r) = edited_range(UNSIGNED, st.free, locked, false, 12.0) {
            locked = r;
            st.stale = false;
        }
        assert_eq!(locked, (0.0, 12.0));
        assert!(!st.stale);
    }
}
