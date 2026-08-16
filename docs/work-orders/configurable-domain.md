# Work order: configurable domain and mesh, 2x largest domain

GOAL
Replace the fixed "Plume" domain tiers (Compact / Standard / Long) with
directly configurable domain extent and mesh resolution, and make the
largest domain roughly 2x the current largest in both directions.

This deliberately REVERSES the physics-reference §8 decision to trade
plume length for throat resolution. At this stage simulation quality
outranks interactivity: a fine mesh in a large domain that takes minutes
to converge is worth more than a coarse mesh that returns instantly and
may not be accurate. §8 records the reversal; the trust caveat about the
downstream diamonds stays word for word.

MEASURED BASELINE (read from the code, not from memory)
The tiers lived in cfd-ui: CaseParams { lz_rt: 46.4, lr_rt: 10.0,
cells_per_rt: 20.0, plume: Compact|Standard|Long }. domain() added the
plume extension: Standard = +8 exit radii and 1.3x radially (the
default, 69.0 x 13 r_t for the demo case, 39,312 graded cells on
record); Long = +28 exit radii and 1.6x radially = 125.6 x 16 r_t for
the demo (57,318 graded cells on record). The new largest is 282 x 32
r_t (2.2x / 2.0x). The task brief quoted the old largest as ~141 r_t
axial; the code says 125.6 — the 282 x 32 target stands either way.

1. RENAME
"Plume" becomes "Domain size". The PlumeLength enum and CaseParams.plume
field are deleted outright — lz_rt / lr_rt now mean the FULL domain
extents, directly configurable, not a compact box plus a hidden tier
extension. References to the exhaust plume as a flow feature (sponge
diagnostic, §9 transit reasoning, shock cells) are NOT renamed.

2. DOMAIN AND MESH ARE FOUR NUMBERS IN THE SIDEBAR (UI only, no config file)
   - cells across the throat radius (base radial resolution): default 20,
     range 8-160. The mass-flow badge goes amber below 20, as before.
   - axial-to-radial aspect dz/dr: default 2.9 (the §8 anisotropy),
     range 1-8. §8 depends on dz >> dr being a free lever; it stays
     exposed. (First lever if T8 disappoints: aspect -> 2.0.)
   - domain length and radius in r_t: default 282 x 32 (the new largest).
   Three preset buttons remain as shortcuts that just fill the fields:
     Preview  46.4 x 10  (the historic compact interactive domain)
     Standard 141 x 16   (half the Large domain each way)
     Large    282 x 32   (= the defaults)
   Engine presets keep sizing their own domain to the bell (the old
   Standard-equivalent extents); the fields stay editable afterwards.
   All four inputs are clamped to their ranges; NaN and non-numeric
   entry cannot reach the solver (egui DragValue plus explicit finite
   checks on commit). Committing a change rebuilds and restarts the
   field, exactly as the tier buttons did.

3. GRADING RULE UNCHANGED IN KIND, EXPLICIT ABOUT THE RADIAL HOLD
Base cells across the geometry, 1.05 growth beyond, 6x width cap —
grade_from_solid is untouched. One thing the tier code did implicitly is
now explicit: the rasterization target extends radially to
min(lr_rt, 3.54 x the wall's outer radius) — 3.54 r_e is the §8 compact
radial sizing (peak plume radius ~2.2 r_e with 1.6x margin) — so base
radial resolution is held across the plume core, not just across the
solid span. For the demo case this reproduces the old held region
(r <= 10) exactly. Beyond it the far field grades out to lr_rt. dt is
set by the held base spacing and does not shrink as the domain grows.

Sponge: still the outer 24 rows, dt-based, sigma_max = 12 a/L_sponge
with PHYSICAL depth. At 282 x 32 the sponge occupies the graded outer
rows (~0.3 r_t wide, capped), entry near r = 25 = 4x the peak plume
radius: clear of the plume edge, and the rows it wastes are cheap 6x
cells. At Preview extents it reproduces the historic entry at r = 8.8.

4. LIVE COST READOUT (above the inputs, updates before any solve)
   - total cell count of the graded grid the settings produce
   - estimated steps to visual steady and estimated wall clock
   - estimated memory
   - amber warning above 30 minutes estimated (Dan's patience budget)
   - a BLOCKING confirmation when the estimate exceeds what the machine's
     memory can hold; the worker also refuses (error message, not a
     crash) as a second line of defence.
Step model: §9 measured ~6,100 steps to visual steady at the 46.4-long
compact demo; steps scale with domain length (transit-dominated) and
with 1/dt via (1/dz + 1/dr). Wall clock uses cells/s measured on THIS
machine by a short timed calibration run at every solver build — never
a constant from another machine. Memory model: ~300 B per padded cell
(solver buffers ~82 B + solid fields + the triple-buffered snapshot
copies), checked against MemAvailable (Linux) / total (macOS).

5. BADGES COMPUTED, NOT QUOTED
   - mass flow: +/-(100/N_throat)% — reproduces the §8 table (+/-5% at
     20, +/-2.5% at 40), amber below 20.
   - thrust and exit-Mach staircase-wall bias: first order in cell size,
     so the recorded 13% / 19% at N_throat = 20 scale as 20/N_throat.

6. EXPLICITLY NOT DOING
No solver numerics changes. No cut cells (separate later job — this
change must not pre-empt it; the staircase-wall bias badges above are
the honest stopgap until then). No config file. No per-axis grading
controls beyond the aspect ratio. No AMR, no local timestepping.

7. TESTING
   - Full acceptance ladder re-run (T0-T8 + graded guards); the wedge
     (T6) and cone (T7) rungs are the guard that physics is untouched.
   - Grid convergence at three resolutions, fixed domain: the smooth-
     advection order test now records N = 100/200/400 for both the
     unlimited and minmod paths; observed orders must match the
     committed pre-change values (1.988 / 1.776 on intel-xeon-4c).
   - Old-default equivalence: the old graded-Standard pipeline is
     replicated inline in a test and must produce a BIT-IDENTICAL Grid
     to the new pipeline at the old-equivalent settings (69.03 x 13,
     20 cells/r_t, dz/dr 2.9), cells = 39,312 as recorded. Identical
     grid + untouched solver => identical numbers. If this test fails,
     stop: something other than sizing changed.
   - Verified while here: dt derives from per-cell widths every step
     (kernel::max_wave_speed); every report integral accumulates in
     f64; floors are non-dimensional state constants (grid-free);
     residual normalization is relative to its own step-10 value
     (grid-free); sponge strength is physical-depth-based (grid-free).

8. REPORT
Cells, steps/s, wall clock, and mass flow / C_f / exit Mach at the new
default (= Large) and at Preview/Standard, recorded to docs/results/
per the existing convention, machine-labelled.
