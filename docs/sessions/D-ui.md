# Session D — `cfd-ui`

**Budget: 105 minutes.**

Build against `MockSolver` from minute one. **Do not wait for the real solver.** It implements the same `Solver` trait, produces a plausible moving `Snapshot` instantly, and swapping it for `EulerSolver` at integration is a one-line change if the trait held.

## Read first

`docs/physics-reference.md` §9 and §13, `docs/contract.md`, `cfd-contract/src/lib.rs`.

## Rules

- You own **`cfd-ui/` and nothing else.**
- You are the **sole owner of the `eframe`/`egui` dependency.** Do not let it spread; every other worktree would pay a 2–5 minute cold build.
- Pinned to **eframe 0.31.1**. Use `App::update`, `CentralPanel`, `SidePanel`. Later versions deprecate `App::update` in favour of `App::ui` and unify the panel types; we pin to avoid the churn.
- Never modify `cfd-contract/`. If it is wrong, print `CONTRACT CHANGE REQUEST:` with the diff and stop.
- SI everywhere in the UI. `Snapshot` and `Report` already arrive dimensional.

## What you build

### 1. Threading

Solver on its own thread. `triple_buffer` for state out, `std::sync::mpsc` for `SolverCommand` in, `ctx.request_repaint()` after each publish, **throttled to 16 ms**.

```rust
loop {
    while let Ok(cmd) = rx_cmd.try_recv() { /* apply */ }
    if paused { sleep(8ms); continue; }
    for _ in 0..turbo { solver.step()?; }
    if last_publish.elapsed() >= Duration::from_millis(16) {
        solver.snapshot_into(tx.input_buffer_mut());
        tx.publish();
        ctx.request_repaint();
        last_publish = Instant::now();
    }
}
```

Triple buffer, not `Arc<Mutex<Snapshot>>` and not a channel. A mutex means the UI holds the lock for the whole colormap pass and stalls the solver. An unbounded channel lets a 400 steps/s solver queue against a 60 fps consumer until you run out of memory; a bounded one blocks the solver. Triple buffer is exactly "latest value wins, never blocks either side."

The 16 ms throttle is load-bearing. Without it a fast solver fires `request_repaint()` hundreds of times a second and the UI thread never gets scheduled.

eframe is reactive by default — it repaints on input events only. Without the explicit wake the sim runs and the screen never updates. Do not use `request_repaint_after` from the UI side; it polls even when nothing changed and burns battery on the M1.

### 2. Field rendering

`ColorImage` + `TextureHandle`, with a 256-entry colormap LUT. **Not** an `egui_wgpu` `CallbackTrait`.

Measured: at 512×256 the callback saves 0.35 ms per frame — 2% of the frame budget — and costs a WGSL shader, a bind group layout, a pipeline, prepare/paint plumbing, and debugging on both Metal and DX12. An R32Float texture is the same 4 bytes per texel as RGBA8, so there is not even an upload saving. Take the 15-minute path.

```rust
self.rgba.resize(nx * ny * 4, 0);
let (lo, hi) = snap.range(kind);
let inv = 255.0 / (hi - lo).max(1e-12);
self.rgba.par_chunks_mut(nx * 4).enumerate().for_each(|(j, row)| {
    for i in 0..nx {
        let v = snap.sample(kind, i, j);
        let k = ((v - lo) * inv).clamp(0.0, 255.0) as usize;
        row[4*i .. 4*i+4].copy_from_slice(&lut[k]);
    }
});
let img = egui::ColorImage::from_rgba_unmultiplied([nx, ny], &self.rgba);
match &mut self.tex {
    Some(t) => t.set(img, egui::TextureOptions::NEAREST),
    None    => self.tex = Some(ctx.load_texture("field", img, egui::TextureOptions::NEAREST)),
}
```

`TextureOptions::NEAREST` so cells are readable — engineers want to see cells. Offer LINEAR as a toggle. Reuse the `rgba` buffer; do not allocate per frame. `TextureHandle::set` reuses the same `TextureId`, so there is no GPU-side allocation churn.

**Fold the schlieren exponential into the LUT index** rather than calling `exp()` per pixel. That is the difference between 0.5 ms and 2.1 ms.

Image row 0 is the **largest** r, which is the opposite of `Grid` indexing. You do that flip; nobody else may. Mirror about the axis for display — the solve is always a half-plane.

### 3. Controls

**Altitude slider.** Sends `SolverCommand::SetAmbient`. It must **never reset the field.** The nozzle interior is supersonic downstream of the throat, so ambient pressure cannot propagate upstream into it — changing altitude re-equilibrates only the plume, which is 1–2 plume transits rather than a full restart. That is the difference between a 12–40 second response and a four-minute one on an M1.

The transient is the demo. The user watches the plume go from over-expanded to under-expanded. Do not block during it, do not interpolate between pre-converged states, do not restart.

Shade the slider track below the separation threshold and put a labeled tick at the crossing point, recomputed when the area ratio changes, so the user sees where the trustworthy range ends **before** dragging.

**Area ratio and chamber pressure sliders.** Clamp the area-ratio slider so cells-across-the-throat-radius never drops below 20, and **display that number next to the thrust readout**. Below 20 the mass-flow error exceeds 5%. *[Superseded by the configurable-domain work order: resolution is now a direct input (8–160 cells/r_t) and the N ≥ 20 rule survives as the amber badge; the displayed-next-to-thrust rule stands.]*

**Turbo control** — 1× / 4× / 16× — running N solver steps per rendered frame, gated by a frame-time budget. Five lines, and it buys most of what local timestepping would without destroying the transient's physical meaning.

**Transport bar:** play, pause, single-step, reset.

**Field selector:** one key per view (schlieren, Mach, pressure, temperature, density), mutually exclusive so the canvas never carries two colormaps.

**Colorbar range locked by default,** with an explicit refit button. An auto-rescaling colorbar makes two frames incomparable and makes a normal transient look like divergence.

### 4. Honesty surface

This is not decoration. It is the difference between a toy that teaches and a toy that misleads.

**Residual meter** — normalized L2 of ∂ρ/∂t — with a green settled indicator below 1e-3. **Thrust, c\*, C_f and Isp greyed out whenever unsettled.** A report from an unconverged field is not a number.

**Separation warning** when `p_e/p_a < max(0.40, (1.88*M_e - 1)^-0.64)`:

> **⚠ SEPARATED FLOW — NOT SIMULATED.** Exit pressure ratio p_e/p_a = 0.28 is below the separation threshold of 0.40. A real nozzle would separate inside the divergent section: the boundary layer would detach and a shock would stand inside the nozzle. **This simulation is inviscid — it has no boundary layer and cannot separate.** Thrust and exit-pressure readouts are not valid in this regime.

**Badges:** mass flow ±5% (amber below 20 cells across the throat) *[now computed live as ±(100/N_throat)% — see physics-reference §13]*; exit pressure and exit Mach "area-averaged, ±4%"; thrust coefficient "inviscid, no divergence correction applied"; drawn-geometry mode "sandbox — qualitative only", since no acceptance test covers an arbitrary user drawing.

**Refuse to display:** Isp in seconds (needs real-gas and chemistry — show C_f relative to the 1-D ideal, which is a ratio and defensible); any efficiency to better than 1%; wall heat flux, shear, skin friction, boundary-layer thickness, wall temperature; separation station as a simulation result; absolute thrust for a named real engine; **anything at all while the floor-activation counter is nonzero** — blank the numeric panel and show "solution invalid."

**One permanent line of screen space:**

```
Inviscid Euler · γ = 1.24 · no boundary layer · no chemistry · no heat transfer
```

**Do not** put a numeric shock-cell-spacing readout on screen. The Prandtl–Pack correlation it would come from is a weakly-imperfectly-expanded linearization and is decorative at high underexpansion.

### 5. Geometry editor

⚠ **Superseded by the parametric-wall work (2026-08).** The original instruction was to consume session C's `Editor` data model behind a trait, stubbing it until C's crate landed. The stub (`StubEditor`) shipped and the swap never happened, so for the life of the project the app edited walls through a polyline model with no validation gate while C's validated one sat unused. Both are now gone, merged into `cfd_geom::FreeformEditor` with the domain bounds injected — neither was a superset: C's had the validation gate and no idea the domain had edges, the app's had the domain clamp, `R_MIN`, `R_MARGIN` and endpoint protection and no gate.

What replaced it: the wall is a **parametric curve** (`cfd_geom::NozzleCurve`) edited through **CAD handles** — nine markers for a bell, six for a cone, one per degree of freedom — and the polyline the solver eats is tessellated from it. The old model made the polyline serve as both the solver's input and the editor's control points, which put a drag handle every ~0.1 r_t and made wall editing unusable. Drawn geometry is now a deliberate MODE (`WallState::Freeform`) reached by a one-way break, not the only representation there is.

The trait (`WallEditor`) survives and both editors implement it. The UI still supplies mouse picking, drag, hover highlight, the world/screen transform and a ghost preview before commit — and now ALL of the pixels: `cfd-geom` returns anchors in r_t and tangent directions only. `cfd-geom` supplies the constraints (a per-handle bisection clamp) and the validation gate.

On commit, send `SolverCommand::SetGeometry`. Rate-limit to pointer-up or one per 100 ms — the flip refill is a multi-pass BFS.

### 6. Canvas navigation

Scroll wheel zooms toward the cursor, not the viewport centre. Middle-drag pans regardless of active tool. Space-drag pans as a temporary override. Two-finger trackpad pan and pinch zoom. Fit-to-domain hotkey. No canvas rotation — a rotated view breaks the mental link to the centreline.

## Done when

`cargo run -p cfd-ui` launches against `MockSolver`, shows a moving field, and the altitude slider, transport bar, field selector, turbo control and geometry editor all mutate visible state. **No real solver required to pass.**

## Known traps

- Writing `App::ui` from memory. You are on 0.31.1; it is `App::update`.
- Allocating a fresh `Vec` for the RGBA buffer every frame.
- Calling `exp()` per pixel for schlieren.
- Letting the altitude slider reset the field. It makes the app feel dead and it is the single most common way this demo fails.
- Adding `wgpu` as a direct dependency. You get the GPU through eframe for free; a direct dependency is a three-minute build cost and a day of surface plumbing for one textured quad.
- Auto-rescaling the colorbar.
