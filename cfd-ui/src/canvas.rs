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

use cfd_contract::{FieldKind, Grid, Snapshot};
use cfd_geom::HandleKind;

use crate::colormap::{lut_for, SOLID_RGBA};
use crate::editor::WallEditor;
use crate::worker::UiFrame;

pub const BG: Color32 = Color32::from_gray(14);
const WALL_COMMITTED: Color32 = Color32::from_rgb(150, 150, 160);
const WALL_GHOST: Color32 = Color32::from_rgb(255, 196, 0);
pub const ACCENT: Color32 = Color32::from_rgb(110, 170, 255);
/// Derived markers (the Bézier control point Q and its control polygon) are
/// drawn in a colour that reads as "annotation", never as "grab me".
const DERIVED: Color32 = Color32::from_rgb(140, 130, 175);
/// How far a tangent handle's dot sits from its anchor, in PIXELS. A wall angle
/// has no natural length, so this offset cannot live in world units: at a fixed
/// world offset the dot would leave the screen on a big nozzle and hide inside
/// the wall on a small one.
const TANGENT_DOT_PX: f32 = 34.0;
/// Pick radius, pixels.
const PICK_PX: f32 = 11.0;

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
    /// The handle under the pointer (hover) or being dragged, and the live
    /// clamp message when a drag has hit a constraint.
    pub active_handle: Option<usize>,
    pub clamp: Option<String>,
    /// Ctrl+click or right-click landed on an editor that refuses point edits
    /// (a parametric wall). The app turns this into the one-way break.
    pub wants_point_edit: bool,
}

pub struct Canvas {
    pub view: Option<View>,
    tex: Option<TextureHandle>,
    rgba: Vec<u8>,
    uploaded: Option<UploadKey>,
    drag_point: Option<usize>,
    hover_point: Option<usize>,
    /// Live clamp message from the handle currently being dragged.
    clamp: Option<String>,
    /// Display-resample LUTs for graded grids: the texture is uniform in
    /// WORLD coordinates (so the image is geometrically true), each texel
    /// gathering its containing cell. Rebuilt when the grid changes; on a
    /// uniform grid the mapping is the identity (one texel per cell).
    lut_grid: Option<Grid>,
    lut_cols: Vec<u32>,
    lut_rows: Vec<u32>,
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
            clamp: None,
            lut_grid: None,
            lut_cols: Vec::new(),
            lut_rows: Vec::new(),
        }
    }

    /// (nx, ny) of the display image; refreshes the cell LUTs if needed.
    fn ensure_luts(&mut self, g: &Grid) -> (usize, usize) {
        if self.lut_grid.as_ref() != Some(g) {
            let (nx, ny) = if g.is_uniform() {
                (g.nz, g.nr) // identity: one texel per cell
            } else {
                // ~1 texel per finest cell, capped to keep uploads bounded.
                (
                    ((g.lz() / g.dz_min() as f64).ceil() as usize).clamp(g.nz, 2048),
                    ((g.lr() / g.dr_min() as f64).ceil() as usize).clamp(g.nr, 1024),
                )
            };
            let lz = g.lz();
            let lr = g.lr();
            self.lut_cols = (0..nx)
                .map(|x| g.z_cell_at((x as f64 + 0.5) * lz / nx as f64) as u32)
                .collect();
            self.lut_rows = (0..ny)
                .map(|y| g.r_cell_at((y as f64 + 0.5) * lr / ny as f64) as u32)
                .collect();
            self.lut_grid = Some(g.clone());
        }
        (self.lut_cols.len(), self.lut_rows.len())
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
        editor: &mut dyn WallEditor,
        editor_on: bool,
        committed_wall: &[[f64; 2]],
        is_mock: bool,
    ) -> CanvasOutput {
        let mut out = CanvasOutput::default();
        // Domain extents come from the snapshot's own grid: presets resize
        // the domain, and the frame is the truth about what is being solved.
        let g = frame.snapshot.grid.clone();
        let (lz, lr) = (g.lz(), g.lr());
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

        // ---- editor interaction. Picking and drag are the UI's job; the
        // backend owns the parameters and their constraints.
        //
        // ALL of it happens in the FOLDED frame. The canvas mirrors the
        // half-plane about the axis and handles are drawn only on the upper
        // copy, so a cursor on the lower copy arrives with negative r. Picking
        // has always folded (`w[1].abs()`); DRAG did not, and an angle handle
        // computed from cursor-minus-anchor inverts on the mirrored copy — drag
        // theta_n on the bottom half and the bell turns inside out. Fold once,
        // here, and pass the folded point to both.
        let space_down = ui.input(|i| i.key_down(egui::Key::Space));
        let mods = ui.input(|i| i.modifiers);
        self.hover_point = None;
        let mut editor_grabbed = false;
        if editor_on {
            let fold = |p: Pos2| {
                let w = view.s2w(rect, p);
                [w[0], w[1].abs()]
            };
            let world = response.hover_pos().map(fold);
            if let Some(w) = world {
                self.hover_point = pick(editor, &view, rect, w);
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
                        self.drag_point = pick(editor, &view, rect, fold(po));
                    }
                }
                if let (Some(i), Some(w), true) = (self.drag_point, world, down) {
                    self.clamp = editor.drag(i, w);
                    editor_grabbed = true;
                }
                // Commit only if the wall actually changed — a motionless
                // click on a handle must not flip the app into a new state.
                if released && self.drag_point.take().is_some() && editor.polyline() != committed_wall
                {
                    out.commit_geometry = true;
                    self.clamp = None;
                }
                if response.clicked() && mods.ctrl {
                    if let Some(w) = world {
                        if editor.insert(w) {
                            out.commit_geometry = true;
                        } else {
                            out.wants_point_edit = true;
                        }
                    }
                }
                if response.secondary_clicked() {
                    match self.hover_point {
                        Some(i) if editor.remove(i) => out.commit_geometry = true,
                        _ => out.wants_point_edit = true,
                    }
                }
            }
        } else {
            self.drag_point = None;
            self.clamp = None;
        }
        if self.drag_point.is_some() {
            editor_grabbed = true;
        }
        out.active_handle = self.drag_point.or(self.hover_point);
        out.clamp = self.clamp.clone();

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
                let iz = g.z_cell_at(z);
                let ir = g.r_cell_at(r);
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
        let g = snap.grid.clone();
        // Uniform-in-world display texels gathered through the cell LUTs: a
        // graded cell occupies exactly its world extent on screen (a plain
        // one-texel-per-cell image would squash the graded tail).
        let (nx, ny) = self.ensure_luts(&g);
        self.rgba.resize(nx * ny * 4, 0);
        let lut = lut_for(field);
        let data = snap.field(field);
        let solid = &snap.solid;
        let (lo, hi) = range;
        let inv = 255.0 / (hi - lo).max(1e-12);
        for j in 0..ny {
            // Image row 0 is the largest r; only cfd-ui flips.
            let ir = self.lut_rows[ny - 1 - j] as usize;
            let src = &data[ir * g.nz..(ir + 1) * g.nz];
            let row = &mut self.rgba[j * nx * 4..(j + 1) * nx * 4];
            for i in 0..nx {
                let iz = self.lut_cols[i] as usize;
                let rgba = if solid.is_solid(ir * g.nz + iz) {
                    SOLID_RGBA
                } else {
                    let k = ((src[iz] - lo) * inv).clamp(0.0, 255.0) as usize;
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
        editor: &dyn WallEditor,
        editor_on: bool,
    ) {
        let to_screen = |pts: &[[f64; 2]], mirror: bool| -> Vec<Pos2> {
            pts.iter()
                .map(|p| view.w2s(rect, [p[0], if mirror { -p[1] } else { p[1] }]))
                .collect()
        };
        let dragging = editor_on && editor.polyline() != committed;
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
                    to_screen(editor.polyline(), mirror),
                    Stroke::new(1.5, WALL_GHOST),
                ));
            }
        }
        if !editor_on {
            return;
        }
        // ---- the dashed N-Q-E control polygon, upper half only. It explains
        // where Q comes from; it is not geometry the user can touch.
        if let Some(poly) = editor.control_polygon() {
            let pts = to_screen(&poly, false);
            for seg in pts.windows(2) {
                dashed(painter, seg[0], seg[1], Stroke::new(1.0, DERIVED));
            }
        }
        // ---- handles, upper half only. Picking folds, so the lower copy is
        // still live; drawing both would double every marker on the axis.
        for (i, h) in editor.handles().iter().enumerate() {
            let pos = handle_screen(h, &view, rect);
            let active = self.drag_point == Some(i);
            let hover = self.hover_point == Some(i);
            match h.kind {
                HandleKind::Derived => {
                    // Q: a hollow diamond, deliberately not a grab dot. It has
                    // zero remaining degrees of freedom — dragging it would
                    // break the G1 tangency at N that makes theta_n mean
                    // anything — so it must not look like the others.
                    let a = view.w2s(rect, h.anchor);
                    let d = 6.0;
                    painter.add(egui::Shape::closed_line(
                        vec![
                            Pos2::new(a.x, a.y - d),
                            Pos2::new(a.x + d, a.y),
                            Pos2::new(a.x, a.y + d),
                            Pos2::new(a.x - d, a.y),
                        ],
                        Stroke::new(1.2, DERIVED),
                    ));
                }
                HandleKind::Tangent { .. } => {
                    // A stalk from the anchor to the dot, so it reads as an
                    // angle about that point rather than a free point.
                    let a = view.w2s(rect, h.anchor);
                    painter.line_segment([a, pos], Stroke::new(1.0, ACCENT.gamma_multiply(0.7)));
                    let (r, fill) = handle_style(active, hover);
                    painter.circle(pos, r, fill, Stroke::new(1.5, ACCENT));
                }
                HandleKind::Point => {
                    let (r, fill) = handle_style(active, hover);
                    painter.circle(pos, r, fill, Stroke::new(1.5, ACCENT));
                }
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

/// Screen position of a handle's marker. Tangent handles sit a FIXED PIXEL
/// distance along their direction, which is why this conversion lives here and
/// not in `cfd-geom`.
fn handle_screen(h: &crate::editor::EditHandle, view: &View, rect: Rect) -> Pos2 {
    let a = view.w2s(rect, h.anchor);
    match h.kind {
        HandleKind::Tangent { dir } => {
            // World (z, r) -> screen (x, y) flips r, and the transform is a
            // uniform scale, so a unit world direction is a unit screen one.
            let (dx, dy) = (dir[0] as f32, -dir[1] as f32);
            let m = dx.hypot(dy).max(1e-6);
            Pos2::new(
                a.x + TANGENT_DOT_PX * dx / m,
                a.y + TANGENT_DOT_PX * dy / m,
            )
        }
        _ => a,
    }
}

fn handle_style(active: bool, hover: bool) -> (f32, Color32) {
    if active {
        (6.0, WALL_GHOST)
    } else if hover {
        (6.0, ACCENT)
    } else {
        (4.0, Color32::from_gray(30))
    }
}

/// Nearest pickable handle within `PICK_PX` of a FOLDED world point, or None.
///
/// Picking is done in screen space because tangent dots are placed in screen
/// space: a world-space hit test would pick the anchor, not the dot the user
/// can see, and the two are a nozzle-size-dependent distance apart.
fn pick(editor: &dyn WallEditor, view: &View, rect: Rect, folded: [f64; 2]) -> Option<usize> {
    let cursor = view.w2s(rect, folded);
    let mut best: Option<(usize, f32)> = None;
    for (i, h) in editor.handles().iter().enumerate() {
        if !h.pickable {
            continue;
        }
        let d = handle_screen(h, view, rect).distance_sq(cursor);
        if d <= PICK_PX * PICK_PX && best.is_none_or(|(_, b)| d < b) {
            best = Some((i, d));
        }
    }
    best.map(|(i, _)| i)
}

/// A dashed segment. egui 0.31 has no dashed stroke on `Painter::line_segment`.
fn dashed(painter: &egui::Painter, a: Pos2, b: Pos2, stroke: Stroke) {
    const DASH: f32 = 6.0;
    const GAP: f32 = 4.0;
    let d = b - a;
    let len = d.length();
    if len < 1e-3 {
        return;
    }
    let step = DASH + GAP;
    let mut t = 0.0;
    while t < len {
        let t1 = (t + DASH).min(len);
        painter.line_segment([a + d * (t / len), a + d * (t1 / len)], stroke);
        t += step;
    }
}
