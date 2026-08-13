//! The field canvas: LUT texture rendering, world/screen transform, pan/zoom,
//! wall overlay and the geometry-editor interaction layer.
//!
//! Conventions owned by this file (docs/sessions/D-ui.md §2, §6):
//! - Image row 0 is the LARGEST r — the opposite of `Grid` indexing. The flip
//!   happens here and nowhere else.
//! - The solve is a half-plane; display mirrors it about the axis by drawing
//!   the same texture twice, the lower copy with flipped UVs (no CPU cost).
//! - Scroll zooms toward the cursor. Middle-drag pans regardless of tool;
//!   space-drag pans as a temporary override. No rotation, ever.

use eframe::egui::{
    self, Color32, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, TextureHandle, TextureOptions,
    Vec2,
};

use cfd_contract::{FieldKind, Snapshot};

use crate::colormap::{lut_for, SOLID_RGBA};
use crate::editor::EditorBackend;
use crate::worker::UiFrame;

pub const BG: Color32 = Color32::from_gray(14);
const WALL_COMMITTED: Color32 = Color32::from_rgb(150, 150, 160);
const WALL_GHOST: Color32 = Color32::from_rgb(255, 196, 0);
const ACCENT: Color32 = Color32::from_rgb(110, 170, 255);

#[derive(Clone, Copy)]
pub struct View {
    /// World point (z, r) at the viewport centre.
    pub center: [f64; 2],
    /// Pixels per throat radius.
    pub scale: f32,
}

impl View {
    /// Fit to the current domain (lz, lr in r_t) — per-case since presets
    /// resize the domain.
    pub fn fit(rect: Rect, lz: f64, lr: f64) -> Self {
        let scale = (rect.width() / lz as f32).min(rect.height() / (2.0 * lr) as f32) * 0.96;
        View {
            center: [lz * 0.5, 0.0],
            scale: scale.max(1.0),
        }
    }

    pub fn w2s(&self, rect: Rect, p: [f64; 2]) -> Pos2 {
        Pos2 {
            x: rect.center().x + ((p[0] - self.center[0]) as f32) * self.scale,
            y: rect.center().y - ((p[1] - self.center[1]) as f32) * self.scale,
        }
    }

    pub fn s2w(&self, rect: Rect, pos: Pos2) -> [f64; 2] {
        [
            self.center[0] + ((pos.x - rect.center().x) / self.scale) as f64,
            self.center[1] - ((pos.y - rect.center().y) / self.scale) as f64,
        ]
    }
}

/// What changed since the last texture upload.
#[derive(PartialEq, Clone, Copy)]
struct UploadKey {
    frame_gen: u64,
    field: FieldKind,
    range: (f32, f32),
    smooth: bool,
}

pub struct HoverInfo {
    pub z_nd: f64,
    pub r_nd: f64,
    /// `None` on a wall cell.
    pub value: Option<f32>,
}

#[derive(Default)]
pub struct CanvasOutput {
    pub hover: Option<HoverInfo>,
    /// The editor changed the wall and the change should be committed
    /// (pointer-up, insert or remove — the rate limit from D-ui.md §5).
    pub commit_geometry: bool,
    /// A primary-button drag was consumed by the space-pan override this
    /// frame (the app uses it to suppress the play/pause toggle on release).
    pub space_panned: bool,
}

pub struct Canvas {
    pub view: Option<View>,
    tex: Option<TextureHandle>,
    rgba: Vec<u8>,
    uploaded: Option<UploadKey>,
    drag_point: Option<usize>,
    hover_point: Option<usize>,
}

impl Canvas {
    pub fn new() -> Self {
        Canvas {
            view: None,
            tex: None,
            rgba: Vec::new(),
            uploaded: None,
            drag_point: None,
            hover_point: None,
        }
    }

    pub fn request_fit(&mut self) {
        self.view = None;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        frame: &UiFrame,
        frame_gen: u64,
        field: FieldKind,
        range: (f32, f32),
        smooth: bool,
        editor: &mut dyn EditorBackend,
        editor_on: bool,
        committed_wall: &[[f64; 2]],
        is_mock: bool,
    ) -> CanvasOutput {
        let mut out = CanvasOutput::default();
        // Domain extents come from the snapshot's own grid: presets resize
        // the domain, and the frame is the truth about what is being solved.
        let g = frame.snapshot.grid;
        let (lz, lr) = (g.nz as f64 * g.dz as f64, g.nr as f64 * g.dr as f64);
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let mut view = *self.view.get_or_insert_with(|| View::fit(rect, lz, lr));

        // ---- zoom toward the cursor (scroll wheel + pinch), never the centre.
        if response.hovered() {
            let (scroll_y, pinch) = ui.input(|i| (i.raw_scroll_delta.y, i.zoom_delta()));
            let factor = pinch * (scroll_y * 0.0015).exp();
            if (factor - 1.0).abs() > 1e-4 {
                if let Some(cursor) = response.hover_pos() {
                    let anchor = view.s2w(rect, cursor);
                    view.scale = (view.scale * factor).clamp(2.0, 4000.0);
                    let after = view.w2s(rect, anchor);
                    let d = after - cursor;
                    view.center[0] += (d.x / view.scale) as f64;
                    view.center[1] -= (d.y / view.scale) as f64;
                }
            }
        }

        // ---- editor interaction (picking and drag are the UI's job; the
        // backend owns the points).
        let space_down = ui.input(|i| i.key_down(egui::Key::Space));
        let mods = ui.input(|i| i.modifiers);
        self.hover_point = None;
        let mut editor_grabbed = false;
        if editor_on {
            let tol = (10.0 / view.scale) as f64;
            let world = response.hover_pos().map(|p| {
                let w = view.s2w(rect, p);
                [w[0], w[1].abs()] // picking works on either mirror half
            });
            if let Some(w) = world {
                self.hover_point = editor.hit_test(w, tol);
            }
            if !space_down {
                // Grab on PRESS, not on egui's drag-start: by the time the
                // drag threshold trips, a fast pointer has already left the
                // hit tolerance and the gesture would fall through to pan.
                // Hit-test at press_origin, not the live pointer — a fast
                // flick delivers press+move in one input batch and the live
                // position has already left the point.
                let (pressed, down, released, press_origin) = ui.input(|i| {
                    (
                        i.pointer.primary_pressed(),
                        i.pointer.primary_down(),
                        i.pointer.primary_released(),
                        i.pointer.press_origin(),
                    )
                });
                if pressed {
                    if let Some(po) = press_origin.filter(|po| rect.contains(*po)) {
                        let w = view.s2w(rect, po);
                        self.drag_point = editor.hit_test([w[0], w[1].abs()], tol);
                    }
                }
                if let (Some(i), Some(w), true) = (self.drag_point, world, down) {
                    editor.drag(i, w);
                    editor_grabbed = true;
                }
                // Commit only if the wall actually changed — a motionless
                // click on a point must not flip the app into sandbox mode.
                if released && self.drag_point.take().is_some() && editor.points() != committed_wall
                {
                    out.commit_geometry = true;
                }
                if response.clicked() && mods.ctrl {
                    if let Some(w) = world {
                        editor.insert(w);
                        out.commit_geometry = true;
                    }
                }
                if response.secondary_clicked() {
                    if let Some(i) = self.hover_point {
                        editor.remove(i);
                        out.commit_geometry = true;
                    }
                }
            }
        } else {
            self.drag_point = None;
        }
        if self.drag_point.is_some() {
            editor_grabbed = true;
        }

        // ---- pan: middle-drag always; primary-drag when space is held or
        // nothing in the editor wants the pointer.
        let mut pan = Vec2::ZERO;
        if response.dragged_by(egui::PointerButton::Middle) {
            pan += response.drag_delta();
        } else if response.dragged_by(egui::PointerButton::Primary) {
            let free = !editor_on || (self.hover_point.is_none() && self.drag_point.is_none());
            if space_down || free {
                pan += response.drag_delta();
                if space_down {
                    out.space_panned = true;
                }
            }
        }
        if pan != Vec2::ZERO && !editor_grabbed {
            view.center[0] -= (pan.x / view.scale) as f64;
            view.center[1] += (pan.y / view.scale) as f64;
        }
        self.view = Some(view);

        // ---- texture upload, only when something actually changed.
        let key = UploadKey {
            frame_gen,
            field,
            range,
            smooth,
        };
        if self.uploaded != Some(key) || self.tex.is_none() {
            self.upload(ui.ctx(), &frame.snapshot, field, range, smooth);
            self.uploaded = Some(key);
        }

        // ---- paint.
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, BG);
        if let Some(tex) = &self.tex {
            let full_uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
            let flipped_uv = Rect::from_min_max(Pos2::new(0.0, 1.0), Pos2::new(1.0, 0.0));
            let upper = Rect::from_two_pos(view.w2s(rect, [0.0, lr]), view.w2s(rect, [lz, 0.0]));
            let lower = Rect::from_two_pos(view.w2s(rect, [0.0, 0.0]), view.w2s(rect, [lz, -lr]));
            painter.image(tex.id(), upper, full_uv, Color32::WHITE);
            painter.image(tex.id(), lower, flipped_uv, Color32::WHITE);
            // Domain outline and centreline.
            let outline = Rect::from_two_pos(view.w2s(rect, [0.0, lr]), view.w2s(rect, [lz, -lr]));
            painter.rect_stroke(
                outline,
                0.0,
                Stroke::new(1.0, Color32::from_gray(70)),
                StrokeKind::Outside,
            );
            painter.line_segment(
                [view.w2s(rect, [0.0, 0.0]), view.w2s(rect, [lz, 0.0])],
                Stroke::new(1.0, Color32::from_white_alpha(14)),
            );
        }

        self.paint_wall(&painter, rect, view, committed_wall, editor, editor_on);
        if is_mock {
            self.paint_watermark(&painter, rect);
        }

        // ---- hover readout.
        if let Some(pos) = response.hover_pos() {
            let w = view.s2w(rect, pos);
            let (z, r) = (w[0], w[1].abs());
            if (0.0..lz).contains(&z) && r < lr {
                let iz = ((z / g.dz as f64) as usize).min(g.nz - 1);
                let ir = ((r / g.dr as f64) as usize).min(g.nr - 1);
                let solid = frame.snapshot.solid.is_solid(g.idx(iz, ir));
                out.hover = Some(HoverInfo {
                    z_nd: z,
                    r_nd: w[1],
                    value: (!solid).then(|| frame.snapshot.sample(field, iz, ir)),
                });
            }
        }
        out
    }

    /// LUT pass into the reused RGBA buffer, then one GPU upload.
    /// `TextureHandle::set` keeps the same `TextureId`, so there is no
    /// GPU-side allocation churn.
    fn upload(
        &mut self,
        ctx: &egui::Context,
        snap: &Snapshot,
        field: FieldKind,
        range: (f32, f32),
        smooth: bool,
    ) {
        let g = snap.grid;
        let (nx, ny) = (g.nz, g.nr);
        self.rgba.resize(nx * ny * 4, 0);
        let lut = lut_for(field);
        let data = snap.field(field);
        let solid = &snap.solid;
        let (lo, hi) = range;
        let inv = 255.0 / (hi - lo).max(1e-12);
        for j in 0..ny {
            let ir = ny - 1 - j; // image row 0 is the largest r; only cfd-ui flips
            let src = &data[ir * nx..(ir + 1) * nx];
            let row = &mut self.rgba[j * nx * 4..(j + 1) * nx * 4];
            for i in 0..nx {
                let rgba = if solid.is_solid(ir * nx + i) {
                    SOLID_RGBA
                } else {
                    let k = ((src[i] - lo) * inv).clamp(0.0, 255.0) as usize;
                    lut[k]
                };
                row[4 * i..4 * i + 4].copy_from_slice(&rgba);
            }
        }
        let img = egui::ColorImage::from_rgba_unmultiplied([nx, ny], &self.rgba);
        // NEAREST by default: engineers want to see cells.
        let opts = if smooth {
            TextureOptions::LINEAR
        } else {
            TextureOptions::NEAREST
        };
        match &mut self.tex {
            Some(t) => t.set(img, opts),
            None => self.tex = Some(ctx.load_texture("field", img, opts)),
        }
    }

    fn paint_wall(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        view: View,
        committed: &[[f64; 2]],
        editor: &dyn EditorBackend,
        editor_on: bool,
    ) {
        let to_screen = |pts: &[[f64; 2]], mirror: bool| -> Vec<Pos2> {
            pts.iter()
                .map(|p| view.w2s(rect, [p[0], if mirror { -p[1] } else { p[1] }]))
                .collect()
        };
        let dragging = editor_on && editor.points() != committed;
        for mirror in [false, true] {
            let color = if dragging {
                Color32::from_rgba_unmultiplied(150, 150, 160, 110)
            } else {
                WALL_COMMITTED
            };
            painter.add(egui::Shape::line(
                to_screen(committed, mirror),
                Stroke::new(1.5, color),
            ));
            if editor_on && dragging {
                // Ghost preview of the uncommitted wall.
                painter.add(egui::Shape::line(
                    to_screen(editor.points(), mirror),
                    Stroke::new(1.5, WALL_GHOST),
                ));
            }
        }
        if editor_on {
            for (i, p) in editor.points().iter().enumerate() {
                let pos = view.w2s(rect, *p);
                let (radius, fill) = if self.drag_point == Some(i) {
                    (6.0, WALL_GHOST)
                } else if self.hover_point == Some(i) {
                    (6.0, ACCENT)
                } else {
                    (4.0, Color32::from_gray(30))
                };
                painter.circle(pos, radius, fill, Stroke::new(1.5, ACCENT));
            }
        }
    }

    fn paint_watermark(&self, painter: &egui::Painter, rect: Rect) {
        let galley = painter.layout_no_wrap(
            "ANALYTIC PREVIEW — NOT A CFD SOLUTION".into(),
            FontId::proportional(22.0),
            Color32::from_white_alpha(26),
        );
        let pos = rect.center() - galley.size() * 0.5;
        painter.add(
            egui::epaint::TextShape::new(pos, galley, Color32::from_white_alpha(26))
                .with_angle(-0.06),
        );
    }
}

/// Compact value formatting for colorbar labels and readouts.
pub fn fmt_value(v: f32) -> String {
    let a = v.abs();
    if v == 0.0 {
        "0".into()
    } else if !(1e-2..1e5).contains(&a) {
        format!("{v:.2e}")
    } else if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 1.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.3}")
    }
}

/// Vertical colorbar strip with min/max labels, painted from the same LUT the
/// texture used. The allocation includes the label rows so nothing clips.
pub fn colorbar(ui: &mut egui::Ui, field: FieldKind, range: (f32, f32), height: f32) {
    const LABEL_H: f32 = 14.0;
    let lut = lut_for(field);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(56.0, height + 2.0 * LABEL_H), Sense::hover());
    let strip = Rect::from_min_max(
        Pos2::new(rect.center().x - 9.0, rect.top() + LABEL_H),
        Pos2::new(rect.center().x + 9.0, rect.bottom() - LABEL_H),
    );
    let painter = ui.painter_at(rect);
    let n = 64;
    for k in 0..n {
        let t0 = k as f32 / n as f32;
        let t1 = (k + 1) as f32 / n as f32;
        let c = lut[(((1.0 - t0) * 255.0) as usize).min(255)];
        let seg = Rect::from_min_max(
            Pos2::new(strip.left(), strip.top() + t0 * strip.height()),
            Pos2::new(strip.right(), strip.top() + t1 * strip.height() + 0.5),
        );
        painter.rect_filled(seg, 0.0, Color32::from_rgb(c[0], c[1], c[2]));
    }
    painter.rect_stroke(
        strip,
        0.0,
        Stroke::new(1.0, Color32::from_gray(90)),
        StrokeKind::Outside,
    );
    painter.text(
        strip.center_top() - Vec2::new(0.0, 3.0),
        egui::Align2::CENTER_BOTTOM,
        fmt_value(range.1),
        FontId::monospace(10.0),
        ui.visuals().text_color(),
    );
    painter.text(
        strip.center_bottom() + Vec2::new(0.0, 3.0),
        egui::Align2::CENTER_TOP,
        fmt_value(range.0),
        FontId::monospace(10.0),
        ui.visuals().text_color(),
    );
}
