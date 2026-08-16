//! The demo case and everything the UI derives from it: the US standard
//! atmosphere behind the altitude slider, the parametric wall contour
//! (an adapter over `cfd_geom::generate_contour` — conical or Rao parabolic
//! bell), and the quasi-1D helpers the honesty surface needs (separation
//! threshold, ideal C_f).
//!
//! All wall geometry here is non-dimensional (units of throat radius); SI
//! appears only in `CaseParams` and the atmosphere. `cfd_geom` speaks SI
//! metres, so `nozzle_contour` converts through `r_throat_m` at the boundary.

use std::sync::Arc;

use cfd_contract::{Ambient, Chamber, GasModel, Grid, Numerics, RefScales, SolidField, SolveSetup};
use cfd_geom::NozzleSpec;

/// The HISTORIC compact demo domain from docs/physics-reference.md §8:
/// 46.4 x 10 r_t at 20 radial cells per throat radius -> the interactive
/// 320 x 200 grid. Still the reference point of the cost model and the
/// "Preview" shortcut; the default domain is now `LZ_DEFAULT x LR_DEFAULT`
/// (§8 reversal — simulation quality outranks interactivity).
pub const LZ: f64 = 46.4;
pub const LR: f64 = 10.0;
pub const CELLS_PER_RT: f64 = 20.0;
/// The §8 anisotropy, dz/dr = 0.1449/0.0500. Widening dz is nearly free in dt
/// (the radial term dominates) while linearly cutting cells. Now a per-case
/// field (`CaseParams::dz_over_dr`); this is its default.
pub const DZ_OVER_DR: f64 = 2.9;
/// Default (and largest-preset) domain extents in throat radii — roughly 2x
/// the old largest tier (the deleted "Long", 125.6 x 16 for the demo case) in
/// both directions. docs/work-orders/configurable-domain.md.
pub const LZ_DEFAULT: f64 = 282.0;
pub const LR_DEFAULT: f64 = 32.0;
/// Radial extent of the held-base-resolution region, as a multiple of the
/// wall's outer radius: §8's compact radial sizing (L_r = 3.54 r_e against a
/// peak plume radius near 2.2 r_e, 1.6x margin). Base radial cells are held
/// through min(lr_rt, PLUME_HOLD * r_wall) so the plume core never lands on
/// graded cells; only the quiescent far field beyond grades out to lr_rt.
pub const PLUME_HOLD: f64 = 3.54;
/// UI input ranges. The §8 N_throat >= 20 rule survives as the amber badge,
/// not as a clamp: 8 cells/r_t is Preview-grade, honestly labelled.
pub const CELLS_PER_RT_RANGE: (f64, f64) = (8.0, 160.0);
pub const DZ_OVER_DR_RANGE: (f64, f64) = (1.0, 8.0);
pub const LZ_RANGE: (f64, f64) = (20.0, 600.0);
pub const LR_RANGE: (f64, f64) = (5.0, 100.0);

/// Altitude ceiling, docs/physics-reference.md §5: 58 km is where the
/// thinnest ambient still clears the pressure floor for the highest-pressure
/// preset (Raptor 2, 300 bar: p_a(58 km) ≈ 27 Pa ≈ 1e-6 p₀). Above the cap
/// the UI switches to the labelled vacuum mode instead.
pub const ALT_MAX_M: f64 = 58_000.0;

/// Fixed back pressure of the vacuum mode, as a fraction of chamber pressure:
/// 30x the positivity floor. The margin is empirical (`vacuum_floor_probe`):
/// the plume expanding past the lip of the biggest bell (Merlin Vac, ε = 165,
/// exit pressure ≈ 8e-5 p₀) undershoots the ambient by roughly an order of
/// magnitude, and at 2x/10x the floor that undershoot lands on the floor and
/// trips the counter (1.1M/0.29M activations in 3000 startup steps), which
/// blanks every readout under the product rule. Atmospheric ambients below
/// this value are clamped to it for the same reason.
pub const VACUUM_P_FRAC: f64 = 30.0 * (cfd_contract::P_MIN_ABS as f64);

/// Universal gas constant, J/(kmol·K).
pub const R_UNIVERSAL_SI: f64 = 8_314.462_618;

/// The separation trigger is effectively the Summerfield 0.40 over the whole
/// demo range — see docs/physics-reference.md §13 (the Schmucker term only
/// participates below M_e = 2.758, via `separation_threshold`).
pub fn separation_threshold(exit_mach: f64) -> f64 {
    if exit_mach <= 0.54 {
        return 0.40; // Schmucker form undefined/meaningless here
    }
    (1.88 * exit_mach - 1.0).powf(-0.64).max(0.40)
}

/// Domain-size shortcuts (docs/work-orders/configurable-domain.md). These are
/// NOT tiers: each one just fills the four editable fields (domain length and
/// radius, cells per throat radius, dz/dr), which stay editable afterwards.
/// The graded tensor-product grid keeps the big domains cheap: base
/// resolution is held across the geometry and the plume core, the far field
/// is covered by geometrically growing cells, and the timestep (set by the
/// finest spacing) does not change at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainPreset {
    /// The historic §8 compact interactive domain: returns in seconds, shows
    /// the Mach disk and half of one shock cell.
    Preview,
    /// Half the Large domain in both directions.
    Standard,
    /// The full domain — the default. §8 reversal: quality over turnaround.
    Large,
}

impl DomainPreset {
    pub const ALL: [DomainPreset; 3] =
        [DomainPreset::Preview, DomainPreset::Standard, DomainPreset::Large];

    pub fn label(self) -> &'static str {
        match self {
            DomainPreset::Preview => "Preview",
            DomainPreset::Standard => "Standard",
            DomainPreset::Large => "Large",
        }
    }

    /// The field values this shortcut fills in:
    /// (lz_rt, lr_rt, cells_per_rt, dz_over_dr).
    pub fn values(self) -> (f64, f64, f64, f64) {
        match self {
            DomainPreset::Preview => (LZ, LR, CELLS_PER_RT, DZ_OVER_DR),
            DomainPreset::Standard => {
                (LZ_DEFAULT / 2.0, LR_DEFAULT / 2.0, CELLS_PER_RT, DZ_OVER_DR)
            }
            DomainPreset::Large => (LZ_DEFAULT, LR_DEFAULT, CELLS_PER_RT, DZ_OVER_DR),
        }
    }
}

/// Diverging-section shape selector. The UI-side twin of
/// `cfd_geom::ContourKind` — the cone half-angle and the bell LENGTH are
/// carried by the case (`CONE_HALF_ANGLE_DEG`, `CaseParams::bell_percent`)
/// and joined back on in `nozzle_contour`; only the measured wall angles,
/// which belong to one engine and to no table, ride on the variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContourKind {
    Conical,
    /// theta_n, theta_e interpolated from the digitised Rao table.
    ParabolicBell,
    /// theta_n, theta_e straight from published geometry — for hardware the
    /// Rao table cannot represent. `bell_percent` still carries the length.
    MeasuredBell { theta_n_deg: f64, theta_e_deg: f64 },
}

impl ContourKind {
    pub fn is_bell(self) -> bool {
        !matches!(self, ContourKind::Conical)
    }
}

/// The §10 parametric family shared by both contours: 30° converging cone,
/// contraction ratio 4, 1.5 r_t upstream arc, 0.382 r_t downstream arc, and
/// the 15° cone half-angle the demo case and the bell's length reference use.
pub const CONE_HALF_ANGLE_DEG: f64 = 15.0;
const CONTRACTION_RATIO: f64 = 4.0;
const CONVERGE_HALF_ANGLE_DEG: f64 = 30.0;
const THROAT_ARC_UP: f64 = 1.5;
/// Downstream throat arc of the §10 family. Per-case since the direct-angle
/// path exists: measured hardware comes with its own arc (Raptor 2, 0.300 r_t).
pub const THROAT_ARC_DOWN: f64 = 0.382;
/// Polyline density for the generated wall. At the coarsest preset grid the
/// longest divergent section (Merlin Vac, ε = 165) spans ~230 axial cells;
/// 256 samples keeps the piecewise-linear wall below the cell scale
/// everywhere while the exact-fraction rasterizer stays cheap.
const CONTOUR_SAMPLES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaseParams {
    pub p0_pa: f64,
    pub t0_k: f64,
    pub gamma: f64,
    pub r_specific_si: f64,
    pub r_throat_m: f64,
    pub area_ratio: f64,
    /// Diverging-section shape; `bell_percent` participates only for the bell
    /// kinds (length as a fraction of the Huzel–Huang 15° reference cone; the
    /// Rao table itself is digitised over 0.6–0.9).
    pub contour_kind: ContourKind,
    pub bell_percent: f64,
    /// Downstream throat-arc radius in r_t. The §10 family value is 0.382;
    /// measured geometry brings its own (Raptor 2: 0.300).
    pub throat_arc_down: f64,
    pub altitude_m: f64,
    /// Labelled vacuum mode: back pressure fixed at `VACUUM_P_FRAC * p0`,
    /// ignoring `altitude_m`. The region above the 58 km slider cap.
    pub vacuum: bool,
    /// FULL domain extents in throat radii and the mesh resolution, all
    /// directly configurable in the sidebar (work order: configurable-domain).
    /// `cells_per_rt` is the base radial resolution (cells across the throat
    /// radius); `dz_over_dr` the §8 anisotropy lever. Engine presets size the
    /// domain to their bell via `preset_domain` (and Merlin Vac drops to
    /// 14 cells/r_t); the demo defaults are the Large domain.
    pub lz_rt: f64,
    pub lr_rt: f64,
    pub cells_per_rt: f64,
    pub dz_over_dr: f64,
}

impl Default for CaseParams {
    fn default() -> Self {
        // The demo case, docs/physics-reference.md §6, on the Large domain.
        CaseParams {
            p0_pa: 5.0e6,
            t0_k: 3200.0,
            gamma: 1.24,
            r_specific_si: 378.0,
            r_throat_m: 0.05,
            area_ratio: 8.0,
            // The demo case is the §10 15° cone; the bell percent is the
            // value a switch to ParabolicBell would use.
            contour_kind: ContourKind::Conical,
            bell_percent: 0.8,
            throat_arc_down: THROAT_ARC_DOWN,
            altitude_m: 0.0,
            vacuum: false,
            lz_rt: LZ_DEFAULT,
            lr_rt: LR_DEFAULT,
            cells_per_rt: CELLS_PER_RT,
            dz_over_dr: DZ_OVER_DR,
        }
    }
}

/// Sanitize the four sidebar fields: clamp into range, replace non-finite
/// entry with the previous committed value. The solver can never see NaN,
/// zero or negative sizing through this path.
pub fn sanitize_domain(p: &mut CaseParams, committed: &CaseParams) {
    let fix = |v: &mut f64, old: f64, range: (f64, f64)| {
        if !v.is_finite() {
            *v = old;
        }
        *v = v.clamp(range.0, range.1);
    };
    fix(&mut p.cells_per_rt, committed.cells_per_rt, CELLS_PER_RT_RANGE);
    fix(&mut p.dz_over_dr, committed.dz_over_dr, DZ_OVER_DR_RANGE);
    fix(&mut p.lz_rt, committed.lz_rt, LZ_RANGE);
    fix(&mut p.lr_rt, committed.lr_rt, LR_RANGE);
}

/// Uniform anisotropic BASE grid over the full extents. At the Preview
/// shortcut values this is exactly the historic 320 x 200 interactive grid.
/// Benchmark/test reference only — the solver runs on `graded_grid`, whose
/// rasterization target is wall-fitted, not this.
#[cfg_attr(not(test), allow(dead_code))]
pub fn base_grid(p: &CaseParams) -> Grid {
    let dr = 1.0 / p.cells_per_rt;
    let dz = p.dz_over_dr * dr;
    Grid::uniform(
        (p.lz_rt / dz).round() as usize,
        (p.lr_rt / dr).round() as usize,
        dz as f32,
        dr as f32,
    )
}

/// The graded solver grid for this case and wall: rasterize the wall on a
/// wall-fitted uniform target, find the solid span, hold base resolution
/// across it and grade geometrically out to the full extents (the spacing
/// rule itself lives in cfd-geom; growth 1.05, width cap 6x).
///
/// The rasterization target spans the wall bounding box plus the grading
/// margin axially, and min(lr_rt, PLUME_HOLD x wall outer radius) radially.
/// `rasterize_wall` closes the solid above the wall to the target's top row,
/// so the radial solid span — and with it the held-base region — reaches that
/// top: base radial cells cover the plume core (§8's 3.54 r_e compact
/// sizing), and only the quiescent far field beyond grades out to lr_rt. For
/// the demo case this reproduces the old pipeline's held region exactly.
/// Keeping the target wall-sized (instead of domain-sized) is what makes the
/// live cost readout cheap at any domain extent.
pub fn graded_grid(p: &CaseParams, wall: &[[f64; 2]]) -> Grid {
    let dr = 1.0 / p.cells_per_rt;
    let dz = p.dz_over_dr * dr;
    let z_wall = wall.iter().map(|q| q[0]).fold(0.0, f64::max);
    let r_wall = wall.iter().map(|q| q[1]).fold(0.0, f64::max);
    let lz_t = (z_wall + 16.0 * dz).min(p.lz_rt);
    let lr_t = (PLUME_HOLD * r_wall).min(p.lr_rt);
    let target = Grid::uniform(
        ((lz_t / dz).round() as usize).max(4),
        ((lr_t / dr).round() as usize).max(4),
        dz as f32,
        dr as f32,
    );
    let solid = rasterize_wall(wall, &target);
    let spec = cfd_geom::GradeSpec::new(dz, dr, p.lz_rt, p.lr_rt);
    cfd_geom::grade_from_solid(&solid, &spec).unwrap_or(target)
}

/// Engine-preset domain sizing in throat radii, (length, radius) — FULL
/// extents. Radius must hold the bell exit plus plume spread; length the
/// nozzle plus a plume that grows with exit radius. These are the same
/// physical extents the old Standard tier produced for a preset, so preset
/// behaviour (and cost) is unchanged by the tier removal.
pub fn preset_domain(area_ratio: f64) -> (f64, f64) {
    let s = area_ratio.sqrt();
    (
        3.0 * (s - 1.0) + 20.0 + 8.0 * s,
        (2.5 * s).max(10.0) * 1.3,
    )
}

/// US standard atmosphere 1976, layers to 71 km (clamped at the 58 km slider
/// cap). Returns (pressure Pa, temperature K).
pub fn atmosphere(h_m: f64) -> (f64, f64) {
    let h = h_m.clamp(0.0, ALT_MAX_M);
    if h < 11_000.0 {
        let t = 288.15 - 0.0065 * h;
        (101_325.0 * (t / 288.15).powf(5.255_88), t)
    } else if h < 20_000.0 {
        let t = 216.65;
        (22_632.06 * (-(h - 11_000.0) / 6_341.62).exp(), t)
    } else if h < 32_000.0 {
        let t = 216.65 + 0.001 * (h - 20_000.0);
        (5_474.889 * (216.65 / t).powf(34.162_6), t)
    } else if h < 47_000.0 {
        let t = 228.65 + 0.0028 * (h - 32_000.0);
        (868.019 * (228.65 / t).powf(12.200_9), t)
    } else if h < 51_000.0 {
        let t = 270.65;
        (110.906 * (-(h - 47_000.0) / 7_922.3).exp(), t)
    } else {
        let t = 270.65 - 0.0028 * (h - 51_000.0);
        (66.939 * (t / 270.65).powf(12.200_9), t)
    }
}

/// Non-dimensional ambient state (chamber-referenced). Vacuum mode fixes the
/// back pressure at `VACUUM_P_FRAC`; the altitude path clamps to the same
/// value so a high-pressure chamber (Raptor 2 above ~52 km) can never push
/// the ambient onto the positivity floor.
pub fn ambient_nd(p: &CaseParams) -> Ambient {
    if p.vacuum {
        let (_, ta) = atmosphere(ALT_MAX_M);
        return Ambient {
            p: VACUUM_P_FRAC as f32,
            t: (ta / p.t0_k) as f32,
        };
    }
    let (pa, ta) = atmosphere(p.altitude_m);
    Ambient {
        p: (pa / p.p0_pa).max(VACUUM_P_FRAC) as f32,
        t: (ta / p.t0_k) as f32,
    }
}

// ---------------------------------------------------------------------------
// Real-engine presets. Numbers researched and verified — do not substitute.
// γ, T₀ and MW are propellant-class values, not measured: no manufacturer
// publishes combustion-gas properties per engine, and the UI labels them so.
// A preset is applied whole (geometry, area ratio, chamber pressure, gas
// model, domain) — never partially.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnginePreset {
    pub name: &'static str,
    pub propellant: &'static str,
    pub area_ratio: f64,
    pub p0_pa: f64,
    pub r_throat_m: f64,
    pub gamma: f64,
    pub t0_k: f64,
    pub mw_g_mol: f64,
    pub cells_per_rt: f64,
    /// Diverging-section shape. Every real engine flies a bell; the demo/
    /// custom case keeps the 15° cone.
    pub contour_kind: ContourKind,
    /// Bell length as a fraction of the Huzel–Huang 15° reference cone (the
    /// Rao table is digitised over 0.6–0.9).
    pub bell_percent: f64,
    /// Downstream throat arc in r_t: the §10 family's 0.382 unless the
    /// engine's own is published (Raptor 2, 0.300).
    pub throat_arc_down: f64,
    /// Where the contour parameters come from, surfaced in the tooltip
    /// exactly like γ/T₀/MW are labelled propellant-class values:
    /// "published" (RS-25 only), "measured geometry" (Raptor 2, whose wall
    /// angles bypass the table entirely), or "design-class estimate".
    pub bell_source: &'static str,
    /// Shown as "(slow)" in the selector: run time ≈12× Merlin 1D.
    pub slow: bool,
    /// Preset-specific honesty note, surfaced as a tooltip.
    pub note: &'static str,
}

pub const PRESETS: [EnginePreset; 6] = [
    EnginePreset {
        name: "Merlin 1D",
        propellant: "LOX/RP-1",
        area_ratio: 16.0,
        p0_pa: 9.7e6,
        r_throat_m: 0.131,
        gamma: 1.24,
        t0_k: 3600.0,
        mw_g_mol: 21.9,
        cells_per_rt: CELLS_PER_RT,
        contour_kind: ContourKind::ParabolicBell,
        throat_arc_down: THROAT_ARC_DOWN,
        bell_percent: 0.78,
        bell_source: "design-class estimate",
        slow: false,
        note: "",
    },
    EnginePreset {
        name: "F-1",
        propellant: "LOX/RP-1",
        area_ratio: 16.0,
        p0_pa: 7.0e6,
        r_throat_m: 0.465,
        // Hard-coded 1.24 on purpose: the F-1 sits 3% inside the separation
        // threshold at sea level BY DESIGN. Let γ drift and that marginal
        // warning flickers in and out, which reads as a bug instead of a
        // deliberate design point.
        gamma: 1.24,
        t0_k: 3600.0,
        mw_g_mol: 21.9,
        cells_per_rt: CELLS_PER_RT,
        contour_kind: ContourKind::ParabolicBell,
        throat_arc_down: THROAT_ARC_DOWN,
        bell_percent: 0.75,
        bell_source: "design-class estimate",
        slow: false,
        note: "Sits 3% inside the separation threshold at sea level by design \
               — γ is hard-coded at 1.24 so the marginal warning is a \
               deliberate design point, not parameter drift.",
    },
    EnginePreset {
        name: "Raptor 2",
        propellant: "LOX/CH4",
        area_ratio: 34.3,
        p0_pa: 3.0e7,
        r_throat_m: 0.115,
        gamma: 1.16,
        t0_k: 3600.0,
        mw_g_mol: 22.0,
        cells_per_rt: CELLS_PER_RT,
        // Raptor 2 does NOT go through the Rao table. Its published geometry
        // (FAA Analysis Report 2019-001b Table 1) is θ_n 32.0°, θ_e 6.0° on a
        // 0.300 R_t downstream throat arc — not the 0.382 R_t of the Rao/TOP
        // construction. At ε = 34.3 the table reproduces θ_n around a 75–76%
        // bell but its θ_e is ~3.5° off there, and no bell percent satisfies
        // both (a 6° exit sits on the 90% row): the table cannot represent
        // this engine, so the wall angles are taken directly. `bell_percent`
        // stays as the LENGTH (two angles and an exit radius do not fix one).
        // Measured in `rao_table_cannot_represent_raptor2`; NOT a γ effect.
        contour_kind: ContourKind::MeasuredBell {
            theta_n_deg: 32.0,
            theta_e_deg: 6.0,
        },
        throat_arc_down: 0.300,
        bell_percent: 0.76,
        bell_source: "measured geometry (FAA AR 2019-001b Table 1)",
        slow: false,
        note: "Wall angles are the published θ_n 32.0° / θ_e 6.0° on a \
               0.300 R_t throat arc, not a Rao-table bell — no bell percent \
               reproduces both angles. The 76% is the length only.",
    },
    EnginePreset {
        name: "AJ10-190",
        propellant: "N2O4/MMH",
        area_ratio: 55.0,
        p0_pa: 8.6e5,
        r_throat_m: 0.073,
        gamma: 1.23,
        t0_k: 3200.0,
        mw_g_mol: 21.5,
        cells_per_rt: CELLS_PER_RT,
        contour_kind: ContourKind::ParabolicBell,
        throat_arc_down: THROAT_ARC_DOWN,
        bell_percent: 0.78,
        bell_source: "design-class estimate",
        slow: false,
        note: "",
    },
    EnginePreset {
        name: "RS-25",
        propellant: "LOX/LH2",
        area_ratio: 69.0,
        p0_pa: 2.06e7,
        r_throat_m: 0.138,
        gamma: 1.20,
        t0_k: 3600.0,
        mw_g_mol: 13.5,
        cells_per_rt: CELLS_PER_RT,
        contour_kind: ContourKind::ParabolicBell,
        throat_arc_down: THROAT_ARC_DOWN,
        bell_percent: 0.80,
        bell_source: "published",
        slow: false,
        note: "Shows a separation warning at sea level: a known conservatism \
               of the criterion, not an error — the real engine runs on the \
               pad.",
    },
    EnginePreset {
        name: "Merlin Vac",
        propellant: "LOX/RP-1",
        area_ratio: 165.0,
        p0_pa: 9.7e6,
        r_throat_m: 0.128,
        gamma: 1.24,
        t0_k: 3600.0,
        mw_g_mol: 21.9,
        // The one preset below the §8 resolution target: at 20 cells/r_t the
        // ε = 165 domain would run ~34× the Merlin 1D case. 14 cells/r_t
        // keeps it usable; the report's N_throat badge goes amber, honestly.
        cells_per_rt: 14.0,
        contour_kind: ContourKind::ParabolicBell,
        throat_arc_down: THROAT_ARC_DOWN,
        bell_percent: 0.75,
        bell_source: "design-class estimate",
        slow: true,
        note: "The costliest preset by far even at the reduced 14 cells per \
               throat radius (the mass-flow badge goes amber for that reason) \
               — see the run-time multiple above. ε = 165 is also past the end \
               of the digitised Rao table (ε ≤ 100): the bell angles clamp to \
               the ε = 100 row — see the amber flag.",
    },
];

impl EnginePreset {
    /// The complete case for this engine. Everything a preset controls is set
    /// here, together — geometry, area ratio, chamber pressure, γ, chamber
    /// temperature, molecular weight and domain. Never apply one partially.
    pub fn case(&self, altitude_m: f64, vacuum: bool) -> CaseParams {
        let (lz_rt, lr_rt) = preset_domain(self.area_ratio);
        CaseParams {
            p0_pa: self.p0_pa,
            t0_k: self.t0_k,
            gamma: self.gamma,
            r_specific_si: R_UNIVERSAL_SI / self.mw_g_mol,
            r_throat_m: self.r_throat_m,
            area_ratio: self.area_ratio,
            contour_kind: self.contour_kind,
            bell_percent: self.bell_percent,
            throat_arc_down: self.throat_arc_down,
            altitude_m,
            vacuum,
            lz_rt,
            lr_rt,
            cells_per_rt: self.cells_per_rt,
            dz_over_dr: DZ_OVER_DR,
        }
    }

    /// Run time relative to Merlin 1D, for tooltips: steps to steady (∝
    /// domain length / dt) times work per step (∝ graded cell count) — the
    /// same model as `estimate_cost`, as a machine-independent ratio.
    pub fn relative_cost(&self) -> f64 {
        let cost = |e: &EnginePreset| {
            let c = e.case(0.0, false);
            // The preset's OWN wall, not the legacy cone: the grading rule
            // reads its held span from the rasterized geometry, and a bell is
            // shorter than the cone at the same area ratio, so costing the
            // cone prices a nozzle this preset never solves.
            let est = estimate_cost(&c, &nozzle_contour(&c).points, None);
            est.steps * est.cells as f64
        };
        cost(self) / cost(&PRESETS[0])
    }
}

/// UI-side area-Mach inversion by bisection, f64. The real Newton one
/// (physics.rs, session B) is not callable from here; this private copy only
/// feeds display estimates (slider shading, ideal C_f), never the solver.
pub fn mach_from_area_ratio(ar: f64, gamma: f64, supersonic: bool) -> f64 {
    if (ar - 1.0).abs() < 1e-9 {
        return 1.0;
    }
    let area = |m: f64| -> f64 {
        let e = (gamma + 1.0) / (2.0 * (gamma - 1.0));
        (1.0 / m) * ((2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * m * m)).powf(e)
    };
    let (mut lo, mut hi) = if supersonic {
        (1.0, 100.0)
    } else {
        (1e-6, 1.0)
    };
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if (area(mid) > ar) == supersonic {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Quasi-1D exit state for the current area ratio: (M_e, p_e/p0).
pub fn ideal_exit(area_ratio: f64, gamma: f64) -> (f64, f64) {
    let m = mach_from_area_ratio(area_ratio.max(1.0), gamma, true);
    let p_ratio = (1.0 + 0.5 * (gamma - 1.0) * m * m).powf(-gamma / (gamma - 1.0));
    (m, p_ratio)
}

/// Ideal thrust coefficient at this operating point (quasi-1D, full flowing).
pub fn ideal_cf(area_ratio: f64, gamma: f64, pa_over_p0: f64) -> f64 {
    let (_, pe_p0) = ideal_exit(area_ratio, gamma);
    let g = gamma;
    let term = 2.0 * g * g / (g - 1.0)
        * (2.0 / (g + 1.0)).powf((g + 1.0) / (g - 1.0))
        * (1.0 - pe_p0.powf((g - 1.0) / g));
    term.max(0.0).sqrt() + (pe_p0 - pa_over_p0) * area_ratio
}

/// Altitude below which the flow would separate (p_e/p_a below the threshold).
/// `None` means the whole 0–40 km range is trustworthy. `Some(ALT_MAX_M)`
/// means the whole range is separated.
pub fn separation_altitude_m(p: &CaseParams) -> Option<f64> {
    let (m_e, pe_p0) = ideal_exit(p.area_ratio, p.gamma);
    let p_e = pe_p0 * p.p0_pa;
    let p_thr = p_e / separation_threshold(m_e);
    if p_thr >= atmosphere(0.0).0 {
        return None; // never separates, even at sea level
    }
    if p_thr <= atmosphere(ALT_MAX_M).0 {
        return Some(ALT_MAX_M); // separated everywhere in range
    }
    // Atmosphere pressure is monotone decreasing in h: bisect.
    let (mut lo, mut hi) = (0.0, ALT_MAX_M);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if atmosphere(mid).0 > p_thr {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

/// A generated wall together with what it ACTUALLY is. The kind is not the
/// request: when `generate_contour` rejects a spec the points are the fallback
/// cone, and `kind` says `Conical` so the status line cannot go on claiming a
/// bell over a cone. `fallback` carries the rejection message for the UI —
/// `eprintln!` is invisible in a windowed app.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedWall {
    pub points: Vec<[f64; 2]>,
    pub kind: ContourKind,
    pub fallback: Option<String>,
}

/// The wall contour for this case as a polyline (z, r) in r_t units, from
/// `cfd_geom::generate_contour` — the §10 15° cone, the Rao parabolic bell, or
/// a measured-angle bell per `contour_kind`. `cfd_geom` speaks SI metres, so
/// the spec is built with `r_throat_m` and the result divided back out.
///
/// Never panics: `generate_contour` validates its spec and a rejected one
/// (hand-set params outside the bell table, degenerate area ratio) falls back
/// to the legacy 15° cone, which is total. The vanished bell is visible; a
/// crashed app is not — but only if the caller reports `kind`, not the request.
pub fn nozzle_contour(p: &CaseParams) -> GeneratedWall {
    let contour = match p.contour_kind {
        ContourKind::Conical => cfd_geom::ContourKind::Conical {
            half_angle_deg: CONE_HALF_ANGLE_DEG,
        },
        ContourKind::ParabolicBell => cfd_geom::ContourKind::ParabolicBell {
            bell_percent: p.bell_percent,
        },
        ContourKind::MeasuredBell {
            theta_n_deg,
            theta_e_deg,
        } => cfd_geom::ContourKind::DirectBell {
            theta_n_deg,
            theta_e_deg,
            length_fraction: p.bell_percent,
        },
    };
    let spec = NozzleSpec {
        throat_radius_m: p.r_throat_m,
        area_ratio: p.area_ratio,
        contraction_ratio: CONTRACTION_RATIO,
        converge_half_angle_deg: CONVERGE_HALF_ANGLE_DEG,
        throat_arc_up: THROAT_ARC_UP,
        throat_arc_down: p.throat_arc_down,
        contour,
    };
    match cfd_geom::generate_contour(&spec, CONTOUR_SAMPLES) {
        Ok(profile) => {
            let inv = 1.0 / p.r_throat_m;
            GeneratedWall {
                points: profile.points.iter().map(|q| [q[0] * inv, q[1] * inv]).collect(),
                kind: p.contour_kind,
                fallback: None,
            }
        }
        Err(e) => GeneratedWall {
            points: fallback_cone(p.area_ratio),
            kind: ContourKind::Conical,
            fallback: Some(e.to_string()),
        },
    }
}

/// Legacy 15° conical contour per docs/physics-reference.md §10, as a sparse
/// control polyline (z, r) in r_t units: chamber wall at r = 2 (contraction
/// ratio 4), 30° converging cone, 1.5 r_t upstream arc, 0.382 r_t downstream
/// arc, straight cone to r_e = sqrt(area_ratio).
///
/// This was the app's only wall generator until `nozzle_contour` above wired
/// in `cfd_geom::generate_contour`; it survives as (a) the infallible
/// fallback for a rejected `NozzleSpec`, and (b) the independent reference
/// the cone-equivalence test compares the new path against.
fn fallback_cone(area_ratio: f64) -> Vec<[f64; 2]> {
    let alpha = 15f64.to_radians();
    let beta = 30f64.to_radians();
    let (r1, r2, r_c) = (1.5, 0.382, 2.0);
    let r_e = area_ratio.max(1.1).sqrt();
    let z_conv = 2.0; // chamber section length
                      // Throat station: converging cone from (z_conv, r_c) tangent to the
                      // upstream arc at angle beta.
    let z_t = z_conv + (r_c - 1.0 - r1 * (1.0 - beta.cos())) / beta.tan() + r1 * beta.sin();

    let mut pts: Vec<[f64; 2]> = Vec::new();
    pts.push([0.0, r_c]);
    pts.push([z_conv, r_c]);
    // Upstream arc, phi from beta down to 0.
    for phi in [beta, beta * 0.5] {
        pts.push([z_t - r1 * phi.sin(), 1.0 + r1 * (1.0 - phi.cos())]);
    }
    pts.push([z_t, 1.0]); // the throat
                          // Downstream arc, phi from 0 up to alpha.
    for phi in [alpha * 0.5, alpha] {
        pts.push([z_t + r2 * phi.sin(), 1.0 + r2 * (1.0 - phi.cos())]);
    }
    // Straight cone to the exit, one midpoint for editing.
    let ra = 1.0 + r2 * (1.0 - alpha.cos());
    let za = z_t + r2 * alpha.sin();
    let l = (r_e - ra) / alpha.tan();
    pts.push([za + 0.5 * l, ra + 0.5 * l * alpha.tan()]);
    pts.push([za + l, r_e]);
    pts
}

/// Test fixture: the legacy sparse cone under its historic name, so the
/// existing tests (and the cone-equivalence test) keep an independent
/// reference that does not go through `cfd_geom`.
#[cfg(test)]
pub fn conical_contour(area_ratio: f64) -> Vec<[f64; 2]> {
    fallback_cone(area_ratio)
}

/// Wall radius at axial station z by linear interpolation, `None` outside the
/// polyline's z range (open plume region). Test-only since the exact
/// rasterizer replaced the band stub.
#[cfg(test)]
pub fn wall_radius(points: &[[f64; 2]], z: f64) -> Option<f64> {
    if points.len() < 2 || z < points[0][0] || z > points[points.len() - 1][0] {
        return None;
    }
    for w in points.windows(2) {
        let ([z0, r0], [z1, r1]) = (w[0], w[1]);
        if z >= z0 && z <= z1 {
            let t = if z1 > z0 { (z - z0) / (z1 - z0) } else { 0.0 };
            return Some(r0 + t * (r1 - r0));
        }
    }
    None
}

/// Exact rasterization via session C's polygon clipper: the solid region is
/// everything at or above the wall polyline, closed past the domain top —
/// the same semantics the solver's report (open-radius throat scan, lip
/// detection) and the T8 rig assume. The old stub laid only a thin band and
/// left fluid above the wall, which made the detected throat span the whole
/// domain and the report integrate ambient cells.
pub fn rasterize_wall(points: &[[f64; 2]], g: &Grid) -> SolidField {
    let mut poly = points.to_vec();
    if poly.len() >= 2 {
        let r_top = poly
            .iter()
            .map(|p| p[1])
            .fold(g.lr(), f64::max)
            + 1.0;
        let (z_first, z_last) = (poly[0][0], poly[poly.len() - 1][0]);
        poly.push([z_last, r_top]);
        poly.push([z_first, r_top]);
    }
    cfd_geom::rasterize_solid_polygon(&poly, g).unwrap_or_else(|e| {
        // Degenerate hand-drawn input: no wall beats a poisoned one, and the
        // vanished nozzle is immediately visible.
        eprintln!("cfd-ui: wall rasterization failed ({e}); solving without a wall");
        SolidField::empty(g.clone())
    })
}

pub fn make_setup(p: &CaseParams, wall: &[[f64; 2]]) -> SolveSetup {
    let g = graded_grid(p, wall);
    let gas = GasModel {
        gamma: p.gamma as f32,
        r_specific_si: p.r_specific_si,
    };
    SolveSetup {
        solid: Arc::new(rasterize_wall(wall, &g)),
        grid: g,
        gas,
        chamber: Chamber { p0: 1.0, t0: 1.0 },
        ambient: ambient_nd(p),
        numerics: Numerics::default(),
        refs: RefScales::from_chamber(p.r_throat_m, p.p0_pa, p.t0_k, &gas),
    }
}

// ---------------------------------------------------------------------------
// Live cost readout (work order: configurable-domain). Every number here is
// an ESTIMATE for sizing decisions, computed before any solve starts, and the
// UI labels it "~". The wall-clock figure uses cells/s measured on THIS
// machine (worker calibration run) — never a constant from another machine.
// ---------------------------------------------------------------------------

/// Estimated bytes per PADDED cell held by the whole app: the solver's
/// buffers (u_old/u_new/u1/rhs 4x16 B + primitives 16 B + mask/solid bools,
/// ~82 B), the solid-fraction fields, and the triple-buffered SI snapshots
/// (8 f32 fields x ~5 live copies), rounded up for slack.
pub const EST_BYTES_PER_CELL: u64 = 300;
/// The solver's own share of the estimate: what a Rebuild must allocate.
pub const EST_SOLVER_BYTES_PER_CELL: u64 = 100;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostEstimate {
    /// Interior cells of the graded grid these settings produce.
    pub cells: usize,
    pub nz: usize,
    pub nr: usize,
    /// Estimated steps to VISUAL steady (§9 criterion).
    pub steps: f64,
    /// Estimated wall-clock seconds; NaN until a throughput sample exists.
    pub seconds: f64,
    /// Estimated total memory (padded cells x EST_BYTES_PER_CELL).
    pub bytes: u64,
}

/// Cost of solving this case to visual steady. Step model: §9 measured
/// ~6,100 steps on the 46.4-long compact demo at dz/dr = 0.145/0.05; steps
/// scale with domain length (transit-dominated) and with 1/dt, and dt scales
/// like 1/(1/dz + 1/dr) at fixed wave speeds (§5). `cells_per_sec` is the
/// measured throughput of this machine, if a sample exists yet.
pub fn estimate_cost(p: &CaseParams, wall: &[[f64; 2]], cells_per_sec: Option<f64>) -> CostEstimate {
    const STEPS_REF: f64 = 6100.0; // §9, at LZ = 46.4, dz = 0.145, dr = 0.05
    let g = graded_grid(p, wall);
    let dr = 1.0 / p.cells_per_rt;
    let dz = p.dz_over_dr * dr;
    let steps =
        STEPS_REF * (g.lz() / LZ) * ((1.0 / dz + 1.0 / dr) / (1.0 / 0.145 + 1.0 / 0.05));
    let padded = (g.nz + 2 * cfd_contract::NG) * (g.nr + 2 * cfd_contract::NG);
    let seconds = match cells_per_sec {
        Some(cps) if cps > 0.0 => steps * g.len() as f64 / cps,
        _ => f64::NAN,
    };
    CostEstimate {
        cells: g.len(),
        nz: g.nz,
        nr: g.nr,
        steps,
        seconds,
        bytes: padded as u64 * EST_BYTES_PER_CELL,
    }
}

/// Best-effort memory budget for the blocking confirmation: MemAvailable on
/// Linux, total memory on macOS, 8 GB when neither can be read. An estimate
/// for a guard rail, not an allocator.
pub fn memory_budget_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    if let Some(kb) = rest.split_whitespace().next().and_then(|v| v.parse::<u64>().ok()) {
                        return kb * 1024;
                    }
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if let Ok(b) = String::from_utf8_lossy(&out.stdout).trim().parse::<u64>() {
                return b;
            }
        }
    }
    8 * 1024 * 1024 * 1024
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atmosphere_is_monotone_and_sane() {
        let (p0, t0) = atmosphere(0.0);
        assert!((p0 - 101_325.0).abs() < 1.0 && (t0 - 288.15).abs() < 0.01);
        let mut last = f64::INFINITY;
        for km in 0..=58 {
            let (p, t) = atmosphere(km as f64 * 1000.0);
            assert!(p < last, "pressure not monotone at {km} km");
            assert!(t > 180.0 && t < 300.0);
            last = p;
        }
        // 11 km tropopause ~22.6 kPa.
        assert!((atmosphere(11_000.0).0 / 22_632.0 - 1.0).abs() < 0.01);
        // Layer joins: 47 km stratopause ~110.9 Pa, 51 km ~66.9 Pa, and the
        // 58 km cap ~27 Pa (the Raptor-2 floor crossing the cap is set by).
        assert!((atmosphere(47_000.0).0 / 110.906 - 1.0).abs() < 0.01);
        assert!((atmosphere(51_000.0).0 / 66.939 - 1.0).abs() < 0.01);
        assert!((atmosphere(58_000.0).0 / 26.8 - 1.0).abs() < 0.02);
    }

    #[test]
    fn preview_base_grid_is_the_reference_320x200() {
        let (lz_rt, lr_rt, cells_per_rt, dz_over_dr) = DomainPreset::Preview.values();
        let p = CaseParams { lz_rt, lr_rt, cells_per_rt, dz_over_dr, ..CaseParams::default() };
        let g = base_grid(&p);
        assert_eq!((g.nz, g.nr), (320, 200));
        assert!((g.dz(0) - 0.145).abs() < 1e-6 && (g.dr(0) - 0.05).abs() < 1e-7);
    }

    /// The default (Large) domain: 282 x 32 r_t — roughly 2x the deleted
    /// largest tier in both directions — at a graded cell count a few times
    /// the historic 64k compact grid, with the base spacing held across the
    /// nozzle and plume core and the finest spacing unchanged (so dt is
    /// unchanged from the compact grid).
    #[test]
    fn default_large_domain_is_graded_and_dt_neutral() {
        let p = CaseParams::default();
        assert_eq!((p.lz_rt, p.lr_rt), (LZ_DEFAULT, LR_DEFAULT));
        let wall = conical_contour(p.area_ratio);
        let g = graded_grid(&p, &wall);
        assert!((g.lz() - p.lz_rt).abs() < 1e-6 && (g.lr() - p.lr_rt).abs() < 1e-6);
        assert!((g.dz_min() as f64 - 0.145).abs() < 1e-3);
        assert!((g.dr_min() as f64 - 0.05).abs() < 1e-7);
        // A uniform grid at these extents would be ~14x the historic compact
        // 64k; grading must keep it within a few multiples.
        assert!(
            g.len() < 3 * 64_000,
            "graded Large is {} cells ({} x {})",
            g.len(),
            g.nz,
            g.nr
        );
        // Base resolution held across the whole nozzle span...
        let z_end = wall.last().unwrap()[0];
        for iz in 0..g.nz {
            if g.z_edges()[iz + 1] <= z_end {
                assert!((g.dz(iz) as f64 - 0.145).abs() < 1e-3, "dz({iz}) = {}", g.dz(iz));
            }
        }
        // ...and base radial cells across the plume core (§8: 3.54 r_e).
        let r_hold = PLUME_HOLD * p.area_ratio.sqrt();
        for ir in 0..g.nr {
            if g.r_edges()[ir + 1] <= r_hold {
                assert!((g.dr(ir) as f64 - 0.05).abs() < 1e-6, "dr({ir}) = {}", g.dr(ir));
            }
        }
        // Graded cells never shrink below base and never exceed the 6x cap
        // by more than the closing stretch.
        for iz in 0..g.nz {
            let d = g.dz(iz) as f64;
            assert!(d > 0.144 && d < 0.145 * 6.0 * 1.1, "dz({iz}) = {d}");
        }
    }

    /// The three shortcut buttons produce the graded grids the §8 table
    /// quotes (demo case). Loose bands: the exact counts are recorded in
    /// docs/results/; this pins the order of magnitude and prints the truth.
    #[test]
    fn domain_presets_produce_expected_grids() {
        let expect = [
            (DomainPreset::Preview, 25_000, 40_000),
            (DomainPreset::Standard, 45_000, 90_000),
            (DomainPreset::Large, 90_000, 190_000),
        ];
        for (opt, lo, hi) in expect {
            let (lz_rt, lr_rt, cells_per_rt, dz_over_dr) = opt.values();
            let p = CaseParams { lz_rt, lr_rt, cells_per_rt, dz_over_dr, ..CaseParams::default() };
            let g = graded_grid(&p, &conical_contour(p.area_ratio));
            println!(
                "{}: {} x {} = {} cells over {:.1} x {:.1} r_t",
                opt.label(), g.nz, g.nr, g.len(), g.lz(), g.lr()
            );
            assert!(
                (lo..hi).contains(&g.len()),
                "{}: {} cells outside [{lo}, {hi})", opt.label(), g.len()
            );
        }
    }

    /// The OLD default (graded Standard tier) reproduced through the new
    /// direct fields must give a BIT-IDENTICAL grid to the old pipeline,
    /// replicated inline here: raster on the compact 46.4 x 10 base grid,
    /// grade to the Standard extents. Identical grid + untouched solver =>
    /// identical numbers; the recorded cell count (39,312 on intel-xeon-4c)
    /// pins it to the committed baseline. If this fails, something other
    /// than sizing changed — stop and report.
    #[test]
    fn old_default_settings_reproduce_the_old_grid() {
        let r_e = 8.0f64.sqrt();
        let old = CaseParams {
            lz_rt: LZ + 8.0 * r_e, // old Standard: compact + 8 exit radii
            lr_rt: LR * 1.3,       // and 1.3x radial headroom
            cells_per_rt: CELLS_PER_RT,
            dz_over_dr: DZ_OVER_DR,
            ..CaseParams::default()
        };
        let wall = conical_contour(old.area_ratio);
        let new_grid = graded_grid(&old, &wall);

        // The old pipeline, verbatim: rasterize on the compact base grid,
        // grade from its solid span to the full extents.
        let dr = 1.0 / old.cells_per_rt;
        let dz = DZ_OVER_DR * dr;
        let compact = Grid::uniform(
            (LZ / dz).round() as usize,
            (LR / dr).round() as usize,
            dz as f32,
            dr as f32,
        );
        assert_eq!((compact.nz, compact.nr), (320, 200));
        let solid = rasterize_wall(&wall, &compact);
        let spec = cfd_geom::GradeSpec::new(dz, dr, old.lz_rt, old.lr_rt);
        let old_grid = cfd_geom::grade_from_solid(&solid, &spec).unwrap();

        assert_eq!(
            old_grid, new_grid,
            "new pipeline must reproduce the old default grid exactly \
             (old {} x {}, new {} x {})",
            old_grid.nz, old_grid.nr, new_grid.nz, new_grid.nr
        );
        assert_eq!(new_grid.len(), 39_312, "recorded graded-Standard cell count");
    }

    #[test]
    fn presets_fit_their_domain_and_cost_sanely() {
        // Costs are computed from the graded-grid model (steps x cells), so
        // assert structure, not brittle constants: Merlin 1D is the unit,
        // cost rises with area ratio among the full-resolution presets, and
        // Merlin Vac is the costliest.
        let costs: Vec<f64> = PRESETS.iter().map(|p| p.relative_cost()).collect();
        println!("relative costs: {costs:?}");
        assert!((costs[0] - 1.0).abs() < 1e-12);
        // F-1 shares Merlin 1D's area ratio and therefore its domain. The
        // costs are no longer bit-equal because the estimate now prices each
        // preset's OWN wall (a 75% bell against Merlin's 78%), and the grading
        // rule reads its held span from that wall — a few percent, not a tier.
        assert!(
            (costs[1] - costs[0]).abs() < 0.05,
            "F-1 shares Merlin's domain: {costs:?}"
        );
        assert!(costs[2] > 1.5 && costs[3] > costs[2] && costs[4] > costs[3],
                "costs not increasing with area ratio: {costs:?}");
        assert!(costs[5] > costs[4] && costs[5] > 5.0, "Merlin Vac not costliest: {costs:?}");
        for p in PRESETS.iter() {
            let c = p.case(0.0, false);
            // Both the preset's bell and the fallback cone must fit: the
            // cone is what a rejected spec degrades to, so it may not
            // overrun the domain either. The cone is the longer of the two.
            assert!(c.contour_kind.is_bell(), "{}: preset is not a bell", p.name);
            let bell = nozzle_contour(&c);
            assert_eq!(bell.fallback, None, "{}: fell back to the cone", p.name);
            for (which, pts) in [("bell", bell.points), ("cone", conical_contour(c.area_ratio))] {
                let end = pts.last().unwrap();
                assert!(
                    end[0] < c.lz_rt,
                    "{} ({which}): nozzle longer than domain",
                    p.name
                );
                assert!(
                    end[1] + 0.4 < c.lr_rt, // exit lip clears the domain top with margin
                    "{} ({which}): exit outside domain",
                    p.name
                );
            }
            // Ambient stays strictly above the positivity floor everywhere
            // on the slider, and in vacuum mode.
            for alt in [0.0, ALT_MAX_M] {
                let a = ambient_nd(&CaseParams { altitude_m: alt, ..c });
                assert!(a.p > cfd_contract::P_MIN_ABS, "{} at {alt} m", p.name);
            }
            let a = ambient_nd(&CaseParams { vacuum: true, ..c });
            assert!((a.p as f64 / VACUUM_P_FRAC - 1.0).abs() < 1e-6);
        }
        // Merlin Vac is the one reduced-resolution preset.
        assert_eq!(PRESETS[5].cells_per_rt, 14.0);
        assert!(PRESETS[5].slow && PRESETS.iter().filter(|p| p.slow).count() == 1);
    }

    #[test]
    fn contour_is_monotone_in_z_with_unit_throat() {
        let pts = conical_contour(8.0);
        for w in pts.windows(2) {
            assert!(w[1][0] > w[0][0], "z not strictly increasing");
        }
        let r_min = pts.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
        assert!((r_min - 1.0).abs() < 1e-12, "throat radius {r_min}");
        assert!((pts.last().unwrap()[1] - 8.0f64.sqrt()).abs() < 1e-9);
        // Throat station ~4.13 r_t (docs §8 domain layout).
        let z_t = pts.iter().find(|p| p[1] == r_min).unwrap()[0];
        assert!((z_t - 4.13).abs() < 0.02, "throat at z = {z_t}");
    }

    /// Rasterization of every wall the app can produce — the legacy cone
    /// fixture AND the two bells, because no preset ships a cone and a
    /// rasterizer test that only ever sees the cone tests a wall nobody runs.
    /// The bells are the real preset walls (RS-25's table bell, Raptor 2's
    /// measured-angle bell), each on its own preset grid.
    #[test]
    fn rasterized_throat_matches_the_contour() {
        let demo = CaseParams::default();
        let rs25 = PRESETS.iter().find(|p| p.name == "RS-25").unwrap().case(0.0, false);
        let raptor = PRESETS.iter().find(|p| p.name == "Raptor 2").unwrap().case(0.0, false);
        let cases: [(&str, CaseParams, Vec<[f64; 2]>); 3] = [
            ("legacy cone", demo, conical_contour(demo.area_ratio)),
            ("RS-25 80% bell", rs25, nozzle_contour(&rs25).points),
            ("Raptor 2 measured bell", raptor, nozzle_contour(&raptor).points),
        ];
        for (name, p, pts) in cases {
            let g = base_grid(&p);
            let s = rasterize_wall(&pts, &g);
            // Narrowest open radius across nozzle columns should be r_t = 1 +- dr.
            let mut r_open_min = f64::INFINITY;
            for iz in 0..g.nz {
                let z = g.z_center(iz) as f64;
                if wall_radius(&pts, z).is_none() {
                    continue;
                }
                if let Some(ir) = (0..g.nr).find(|&ir| s.is_solid(g.idx(iz, ir))) {
                    r_open_min = r_open_min.min(g.r_face(ir) as f64);
                }
            }
            assert!(
                (r_open_min - 1.0).abs() <= g.dr(0) as f64,
                "{name}: throat {r_open_min}"
            );
            // The wall is actually there: solid cells exist, and the exit lip
            // column is open out to the exit radius (a bell that rasterized as
            // a plug would still pass the throat check above).
            assert!(
                s.fraction.iter().any(|&f| f > 0.0),
                "{name}: nothing rasterized"
            );
            let z_exit = pts.last().unwrap()[0];
            let iz = (0..g.nz)
                .rev()
                .find(|&iz| (g.z_center(iz) as f64) < z_exit)
                .unwrap();
            let open = (0..g.nr).find(|&ir| s.is_solid(g.idx(iz, ir))).unwrap();
            assert!(
                (g.r_face(open) as f64 - p.area_ratio.sqrt()).abs() <= 2.0 * g.dr(0) as f64,
                "{name}: exit-plane open radius {} vs r_e {}",
                g.r_face(open),
                p.area_ratio.sqrt()
            );
        }
    }

    /// The grading rule reads its hold region from the RASTERIZED wall, so it
    /// has to be exercised on a bell: a bell is shorter than the cone and its
    /// exit lip sits at a different station, which is exactly what sets the
    /// held span. Both bell kinds, at their preset domains.
    #[test]
    fn graded_grid_holds_base_across_a_preset_bell() {
        for name in ["RS-25", "Raptor 2"] {
            let pre = PRESETS.iter().find(|p| p.name == name).unwrap();
            let p = pre.case(0.0, false);
            let wall = nozzle_contour(&p);
            assert_eq!(wall.fallback, None, "{name}: fell back to the cone");
            assert!(wall.kind.is_bell());
            let g = graded_grid(&p, &wall.points);
            let (lz, lr) = (p.lz_rt, p.lr_rt);
            assert!((g.lz() - lz).abs() < 1e-6 && (g.lr() - lr).abs() < 1e-6, "{name}: extents");
            let dr = 1.0 / p.cells_per_rt;
            let dz = p.dz_over_dr * dr;
            // Finest spacing unchanged (dt is set by it) — to within the
            // wall-fitted hold's integer-cell rounding, which can land base a
            // fraction of a percent low (RS-25: 0.049994 against 0.05). That
            // direction only ever shrinks dt, so dt-neutrality is preserved.
            assert!(
                g.dr_min() as f64 <= dr + 1e-9 && g.dr_min() as f64 > dr * 0.999,
                "{name}: dr_min {} vs base {dr}",
                g.dr_min()
            );
            assert!((g.dz_min() as f64 - dz).abs() < 1e-3, "{name}: dz_min {}", g.dz_min());
            // … base resolution held across the whole bell in z. (Only in z:
            // `rasterize_wall` closes the solid past the top of the domain, so
            // the r solid span is the full height and the r-hold assert would
            // be unfalsifiable. `dr_min` above is the r-axis check that bites.)
            let z_end = wall.points.last().unwrap()[0];
            for iz in 0..g.nz {
                if g.z_edges()[iz + 1] <= z_end {
                    assert!((g.dz(iz) as f64 - dz).abs() < 1e-3, "{name}: dz({iz}) = {}", g.dz(iz));
                }
            }
            // … and the graded tail actually saves cells against uniform.
            let uniform = (lz / dz).round() as usize * (lr / dr).round() as usize;
            assert!(g.len() < uniform, "{name}: graded {} vs uniform {uniform}", g.len());
            // The bell rasterizes onto the GRADED grid — the grid the solver
            // actually runs on, not the base grid — with an open throat of the
            // right radius. Non-emptiness alone would pass for a solid plug,
            // which is exactly what a mis-wound polygon produces.
            let s = rasterize_wall(&wall.points, &g);
            assert!(s.fraction.iter().any(|&f| f > 0.0), "{name}: nothing rasterized");
            assert!(s.fraction.iter().any(|&f| f < 1.0), "{name}: everything is solid");
            let mut r_open_min = f64::INFINITY;
            for iz in 0..g.nz {
                if (g.z_center(iz) as f64) > z_end {
                    break;
                }
                if let Some(ir) = (0..g.nr).find(|&ir| s.is_solid(g.idx(iz, ir))) {
                    r_open_min = r_open_min.min(g.r_face(ir) as f64);
                }
            }
            assert!(
                (r_open_min - 1.0).abs() <= g.dr_min() as f64,
                "{name}: graded-grid throat {r_open_min}"
            );
        }
    }

    /// The fallback path itself, which no reachable slider setting produces
    /// (every shipped preset and every ε on the slider generates cleanly) and
    /// which therefore only a test can exercise: a rejected spec must come
    /// back labelled as the cone it actually drew, carrying the reason. This
    /// is the assertion the old `Vec<[f64; 2]>` return type could not make —
    /// the status line went on saying "80% bell" over the fallback cone.
    #[test]
    fn a_rejected_spec_reports_the_cone_it_actually_drew() {
        // Bell percent below the digitised table: rejected by NozzleSpec::validate.
        let p = CaseParams {
            contour_kind: ContourKind::ParabolicBell,
            bell_percent: 0.5,
            ..CaseParams::default()
        };
        let w = nozzle_contour(&p);
        assert_eq!(w.kind, ContourKind::Conical, "produced kind must be what was drawn");
        assert!(w.fallback.is_some(), "the reason must reach the caller");
        assert!(
            w.fallback.as_deref().unwrap().contains("bell_percent"),
            "reason should name the offending parameter: {:?}",
            w.fallback
        );
        assert_eq!(w.points, conical_contour(p.area_ratio), "must BE the fallback cone");
        // Same for a measured-angle bell whose angles cannot close a contour.
        let p = CaseParams {
            contour_kind: ContourKind::MeasuredBell {
                theta_n_deg: 6.0,
                theta_e_deg: 32.0,
            },
            ..CaseParams::default()
        };
        let w = nozzle_contour(&p);
        assert_eq!(w.kind, ContourKind::Conical);
        assert!(w.fallback.is_some());
        // And a spec that IS accepted must never claim a fallback.
        let w = nozzle_contour(&CaseParams::default());
        assert_eq!(w.kind, ContourKind::Conical);
        assert_eq!(w.fallback, None);
    }

    #[test]
    fn demo_case_never_separates_in_range() {
        // docs §6: the whole default slider range is trustworthy.
        assert_eq!(separation_altitude_m(&CaseParams::default()), None);
        // A big over-expanded bell at low chamber pressure must separate somewhere.
        let p = CaseParams {
            area_ratio: 16.0,
            p0_pa: 1.0e6,
            ..Default::default()
        };
        let h = separation_altitude_m(&p);
        assert!(
            h.is_some() && h.unwrap() > 0.0,
            "expected a crossing, got {h:?}"
        );
    }

    #[test]
    fn ideal_exit_matches_reference_gamma_124_eps_8() {
        // docs §6 table: M_e = 3.224, p_e = 76.2 kPa at p0 = 5 MPa.
        let (m, pr) = ideal_exit(8.0, 1.24);
        assert!((m - 3.224).abs() < 0.005, "M_e {m}");
        assert!(
            (pr * 5.0e6 / 76_200.0 - 1.0).abs() < 0.01,
            "p_e {}",
            pr * 5.0e6
        );
    }
}

/// The contour-generator swap: cfd_geom::generate_contour replacing the
/// legacy in-crate cone. Results go to docs/results/contour-<machine>.json
/// (CLAUDE.md: results get committed, not reported in chat).
#[cfg(test)]
mod contour_swap {
    use super::*;

    const SUITE: &str = "contour";

    /// Wall slope of the polyline's last segment, as an angle in degrees.
    fn exit_angle_deg(pts: &[[f64; 2]]) -> f64 {
        let n = pts.len();
        let (a, b) = (pts[n - 2], pts[n - 1]);
        ((b[1] - a[1]) / (b[0] - a[0])).atan().to_degrees()
    }

    /// Steepest wall angle downstream of the throat, in degrees — θ_n for a
    /// bell, which peaks exactly at the throat-arc/Bézier junction. Central
    /// differences: the junction vertex straddles two pieces that are both
    /// tangent to θ_n there, so the chord through its neighbours recovers the
    /// angle to second order (a one-sided chord would read ~0.5° low).
    fn max_wall_angle_deg(pts: &[[f64; 2]]) -> f64 {
        let t = throat_index(pts);
        (t + 1..pts.len() - 1)
            .map(|i| {
                let (a, b) = (pts[i - 1], pts[i + 1]);
                ((b[1] - a[1]) / (b[0] - a[0])).atan().to_degrees()
            })
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Index of the throat (minimum-radius) vertex.
    fn throat_index(pts: &[[f64; 2]]) -> usize {
        pts.iter()
            .enumerate()
            .min_by(|a, b| a.1[1].partial_cmp(&b.1[1]).unwrap())
            .unwrap()
            .0
    }

    /// The gate for everything else: the new path with `Conical` must BE the
    /// old cone. Every vertex of the sparse legacy polyline is either an
    /// exact sample point of the dense generated one (arc endpoints and
    /// midpoints land on the shared φ-grid) or lies on one of its straight
    /// segments, so linear interpolation on the new polyline must reproduce
    /// the old vertices to float roundoff. 1e-9 r_t is ~5 decades above
    /// roundoff and ~7 below a cell.
    #[test]
    fn cone_equivalence_new_path_vs_legacy() {
        let mut worst = 0.0f64;
        // Every shipped preset's (ε, r_t), read from PRESETS rather than
        // copied — a test that hard-codes a preset's numbers stops testing the
        // preset the day it changes — plus the demo case and the ε slider's
        // bottom stop.
        let settings: Vec<(f64, f64)> = [(2.0, 0.05), (8.0, 0.05)]
            .into_iter()
            .chain(PRESETS.iter().map(|p| (p.area_ratio, p.r_throat_m)))
            .collect();
        for (eps, rt) in settings {
            let p = CaseParams {
                area_ratio: eps,
                r_throat_m: rt,
                contour_kind: ContourKind::Conical,
                ..CaseParams::default()
            };
            let new = nozzle_contour(&p).points;
            let old = conical_contour(eps);
            // Same span in z…
            worst = worst.max((new[0][0] - old[0][0]).abs());
            let (ne, oe) = (new.last().unwrap(), old.last().unwrap());
            worst = worst.max((ne[0] - oe[0]).abs()).max((ne[1] - oe[1]).abs());
            // …and every legacy vertex on the new wall. The legacy exit z can
            // land an ulp past the new polyline's (algebraically identical)
            // exit z, where wall_radius returns None — clamp into the new
            // span before sampling. Nothing is hidden: the endpoint z's were
            // compared against the same tolerance above.
            let (z_lo, z_hi) = (new[0][0], new.last().unwrap()[0]);
            for q in &old {
                let r = wall_radius(&new, q[0].clamp(z_lo, z_hi))
                    .expect("legacy vertex outside new wall");
                worst = worst.max((r - q[1]).abs());
            }
        }
        cfd_results::record_test(SUITE, cfd_results::TestResult {
            id: "cone-equivalence".into(),
            name: "generate_contour(Conical) vs legacy cone at eps 2, 8 and every preset".into(),
            expected: "<= 1e-9".into(),
            actual: worst.into(),
            units: "max |dr| (r_t)".into(),
            pass: worst <= 1e-9,
        });
        assert!(worst <= 1e-9, "worst deviation {worst:.3e} r_t");
    }

    /// First contact between the bell generator and published hardware:
    /// RS-25, the one preset whose bell percent is actually published (80%).
    /// Nozzle length 121 in on a 10.88 in throat diameter -> 11.12 diameters;
    /// exit wall angle ≈ 7°.
    #[test]
    fn rs25_bell_matches_published_geometry() {
        let pre = PRESETS.iter().find(|p| p.name == "RS-25").unwrap();
        let c = pre.case(0.0, false);
        let pts = nozzle_contour(&c).points;
        let z_t = pts[throat_index(&pts)][0];
        let len_dia = (pts.last().unwrap()[0] - z_t) / 2.0; // r_t -> throat diameters
        let published_dia = 121.0 / 10.88; // 11.12
        let len_err = (len_dia / published_dia - 1.0).abs();
        cfd_results::record_test(SUITE, cfd_results::TestResult {
            id: "rs25-bell-length".into(),
            name: "RS-25 80% bell nozzle length vs published 121 in / 10.88 in".into(),
            expected: "11.12 +/- 2%".into(),
            actual: len_dia.into(),
            units: "throat diameters".into(),
            pass: len_err <= 0.02,
        });
        let exit_deg = exit_angle_deg(&pts);
        cfd_results::record_test(SUITE, cfd_results::TestResult {
            id: "rs25-bell-exit-angle".into(),
            name: "RS-25 80% bell exit wall angle vs published ~7 deg".into(),
            expected: "7 +/- 1".into(),
            actual: exit_deg.into(),
            units: "deg".into(),
            pass: (exit_deg - 7.0).abs() <= 1.0,
        });
        assert!(len_err <= 0.02, "nozzle length {len_dia:.3} dia vs {published_dia:.3}");
        assert!((exit_deg - 7.0).abs() <= 1.0, "exit angle {exit_deg:.2} deg");
    }

    /// Raptor 2 against its published geometry (FAA Analysis Report
    /// 2019-001b Table 1: θ_n 32.0°, θ_e 6.0°, downstream throat arc
    /// 0.300 R_t). Measured off the wall the APP actually builds — through
    /// `nozzle_contour`, at the shipped preset, no hand-copied parameters —
    /// because a test that calls `rao_angles` directly cannot tell you what
    /// the app draws.
    ///
    /// Both angles are asserted now: Raptor 2 no longer goes through the Rao
    /// table. `rao_table_cannot_represent_raptor2` below is the standing
    /// record of why it must not.
    #[test]
    fn raptor2_bell_angles_vs_published() {
        let pre = PRESETS.iter().find(|p| p.name == "Raptor 2").unwrap();
        let wall = nozzle_contour(&pre.case(0.0, false));
        assert_eq!(wall.fallback, None, "Raptor 2 fell back to the cone");
        let (tn, te) = (
            max_wall_angle_deg(&wall.points),
            exit_angle_deg(&wall.points),
        );
        for (id, name, actual, want) in [
            ("raptor2-theta-n", "Raptor 2 measured-geometry theta_n vs FAA AR 2019-001b 32.0 deg", tn, 32.0),
            ("raptor2-theta-e", "Raptor 2 measured-geometry theta_e vs FAA AR 2019-001b 6.0 deg", te, 6.0),
        ] {
            cfd_results::record_test(SUITE, cfd_results::TestResult {
                id: id.into(),
                name: name.into(),
                expected: format!("{want:.1} +/- 0.5").as_str().into(),
                actual: actual.into(),
                units: "deg".into(),
                pass: (actual - want).abs() <= 0.5,
            });
        }
        assert!((tn - 32.0).abs() <= 0.5, "theta_n {tn:.2} deg");
        assert!((te - 6.0).abs() <= 0.5, "theta_e {te:.2} deg");
    }

    /// Why Raptor 2 bypasses the Rao table, kept as an executable record: at
    /// ε = 34.3 NO bell percent in the digitised range reproduces both
    /// published angles. θ_n alone is fine (75.6% lands 0.16° off), θ_e is
    /// off by 3.47° there and only reaches 6° on the 90% row, where θ_n is
    /// wrong by ~3°. Do not "fix" this by re-fitting a bell percent.
    ///
    /// Not a γ effect, and the note says so: Rao (1958) has γ barely moving
    /// the contour at fixed length and area ratio, and the sign is inverted —
    /// Raptor's γ 1.16 against the chart's ~1.23 makes the exit angle LARGER,
    /// not smaller, by well under 0.7°. The gap is 3.47° and points the other
    /// way; it is a contour-family mismatch, not a gas-property one.
    #[test]
    fn rao_table_cannot_represent_raptor2() {
        let pre = PRESETS.iter().find(|p| p.name == "Raptor 2").unwrap();
        // Sweep the whole digitised range: the percent that best satisfies
        // BOTH angles (worst-of-the-two error), and the percent that matches
        // theta_n alone — where theta_e is then free to be as wrong as it is.
        let (mut best_bp, mut best_err) = (f64::NAN, f64::INFINITY);
        let (mut tn_bp, mut tn_err, mut te_there) = (f64::NAN, f64::INFINITY, f64::NAN);
        for k in 0..=300 {
            let bp = 0.6 + 0.3 * k as f64 / 300.0;
            let (tn, te) = cfd_geom::rao_angles(pre.area_ratio, bp);
            if (tn - 32.0).abs().max((te - 6.0).abs()) < best_err {
                best_err = (tn - 32.0).abs().max((te - 6.0).abs());
                best_bp = bp;
            }
            if (tn - 32.0).abs() < tn_err {
                tn_err = (tn - 32.0).abs();
                tn_bp = bp;
                te_there = te;
            }
        }
        cfd_results::record_note(SUITE, "raptor2-rao-table-mismatch", &format!(
            "Raptor 2 is NOT routed through the Rao table. Published geometry (FAA \
             Analysis Report 2019-001b Table 1): theta_n 32.0 deg, theta_e 6.0 deg, \
             downstream throat arc 0.300 R_t (the Rao/TOP construction's 0.382 R_t \
             does not apply). At eps 34.3 the table matches theta_n at {tn_bp:.3} \
             ({tn_err:.2} deg off) but theta_e is {:.2} deg off there, and the percent \
             minimising the WORSE of the two angles ({best_bp:.3}) still misses by \
             {best_err:.2} deg — no bell \
             percent satisfies both, so the table cannot represent this engine. This \
             is a contour-family mismatch, NOT a gamma effect: Rao (1958) states gamma \
             barely moves the contour at fixed length and area ratio, and the sign is \
             inverted (gamma 1.16 vs the chart's ~1.23 makes the exit angle larger, \
             worth under 0.7 deg). The app now takes the two angles directly \
             (ContourKind::DirectBell); 76% remains the LENGTH only.",
            (te_there - 6.0).abs()));
        assert!(
            best_err > 1.0,
            "a {best_bp:.3} bell reproduces both published angles to {best_err:.2} deg \
             — the direct-angle path is no longer needed for Raptor 2"
        );
    }

    /// Every preset's production wall is the bell (not the silent fallback
    /// cone): the generator reports the bell kind it was asked for, the wall
    /// is strictly shorter than the 15° cone at the same area ratio, and its
    /// exit angle is well under 15°.
    #[test]
    fn presets_actually_get_bells() {
        for pre in &PRESETS {
            let c = pre.case(0.0, false);
            let wall = nozzle_contour(&c);
            // The kind PRODUCED, not the kind requested: a rejected spec
            // silently returns the fallback cone, and this is the assert that
            // catches it (the status line reads the same field).
            assert_eq!(
                wall.fallback, None,
                "{}: contour generation fell back to the cone",
                pre.name
            );
            assert!(wall.kind.is_bell(), "{}: produced {:?}", pre.name, wall.kind);
            let bell = wall.points;
            let cone = conical_contour(c.area_ratio);
            assert!(
                bell.last().unwrap()[0] < cone.last().unwrap()[0] - 1.0,
                "{}: wall is not shorter than the cone — fallback engaged?",
                pre.name
            );
            let exit_deg = exit_angle_deg(&bell);
            assert!(
                exit_deg < 12.0,
                "{}: exit angle {exit_deg:.1} deg is not a bell",
                pre.name
            );
        }
    }
}

#[cfg(test)]
mod vacuum_probe {
    use super::*;
    use cfd_contract::Solver;

    /// Worst case for the floors: the biggest bell (Merlin Vac, ε = 165)
    /// starting into the vacuum-mode back pressure. Passes only when the
    /// floor counter stays exactly zero — a nonzero counter blanks the whole
    /// report by the product rule, so a vacuum mode that trips it is not a
    /// feature. ~2 min at the reduced 14 cells/r_t; run explicitly.
    #[test]
    #[ignore = "slow probe: cargo test -p cfd-ui vacuum_floor_probe -- --include-ignored --nocapture"]
    fn vacuum_floor_probe() {
        // Cold starts into the vacuum back pressure. Measured at 3e-5 p0:
        // Merlin 1D is clean even at 2e-6; Merlin Vac (ε 165, exit pressure
        // ≈ 8e-5 p0) always brushes the floor during the startup front (150k
        // activations, the last at step 796; steady state clean), which the
        // cumulative counter turns into a permanent SOLUTION INVALID under
        // the product rule. The interactive path — light at sea level, then
        // move to vacuum via set_ambient, which never resets the field — is
        // asserted clean below; the §5 cell-level first-order redo is the
        // documented cure for the cold-start case and is not implemented.
        for pi in 0..PRESETS.len() {
            for vac in [false, true] {
                let p = PRESETS[pi].case(0.0, vac);
                let setup = make_setup(&p, &conical_contour(p.area_ratio));
                let mut s = cfd_core::EulerSolver::new(setup.clone()).unwrap();
                let (mut floors, mut last_step) = (0, 0);
                for step in 0..2000u64 {
                    let f = s.step().unwrap().floor_activations;
                    if f > floors {
                        last_step = step;
                    }
                    floors = f;
                }
                println!(
                    "vacuum probe: {} cold start ({}) -> floors {floors}, last at step {last_step}",
                    PRESETS[pi].name,
                    if vac { "vacuum" } else { "sea level" }
                );
            }
        }
        // The altitude-slider path for the extreme bell: sea level, settle,
        // then step to the vacuum back pressure without resetting the field.
        let p = PRESETS[5].case(0.0, false);
        let setup = make_setup(&p, &conical_contour(p.area_ratio));
        let mut s = cfd_core::EulerSolver::new(setup).unwrap();
        for _ in 0..1500 {
            s.step().unwrap();
        }
        let at_sl = s.report(); // not asserted; just forces the borrow pattern
        let _ = at_sl;
        let before = {
            let mut f = 0;
            for _ in 0..1 {
                f = s.step().unwrap().floor_activations;
            }
            f
        };
        s.set_ambient(cfd_contract::Ambient {
            p: VACUUM_P_FRAC as f32,
            t: (atmosphere(ALT_MAX_M).1 / p.t0_k) as f32,
        });
        let mut after = before;
        for _ in 0..3000 {
            after = s.step().unwrap().floor_activations;
        }
        println!(
            "vacuum probe: Merlin Vac sea level -> vacuum transition: floors {before} before, {after} after"
        );
        assert_eq!(
            after, before,
            "the altitude-slider path into vacuum must not trip the floors"
        );
    }
}

#[cfg(test)]
mod perf_probe {
    use super::*;
    use cfd_contract::Solver;

    #[test]
    fn snapshot_timing_probe() {
        // Preview extents: the probe times snapshot cost, not domain size.
        let (lz_rt, lr_rt, cells_per_rt, dz_over_dr) = DomainPreset::Preview.values();
        let p = CaseParams { lz_rt, lr_rt, cells_per_rt, dz_over_dr, ..CaseParams::default() };
        let setup = make_setup(&p, &conical_contour(p.area_ratio));
        let mut s = cfd_core::MockSolver::new(setup).unwrap();
        for _ in 0..200 {
            s.step().unwrap();
        }
        let t0 = std::time::Instant::now();
        let n = 5;
        for _ in 0..n {
            std::hint::black_box(s.snapshot());
        }
        eprintln!("snapshot avg: {:?}", t0.elapsed() / n);
        let t0 = std::time::Instant::now();
        for _ in 0..1000 {
            s.step().unwrap();
        }
        eprintln!("1000 steps: {:?}", t0.elapsed());
    }
}

#[cfg(test)]
mod grading_bench {
    use super::*;
    use cfd_contract::Solver;
    use std::time::Instant;

    fn bench_setup(grid: Grid, p: &CaseParams, wall: &[[f64; 2]]) -> SolveSetup {
        let gas = GasModel { gamma: p.gamma as f32, r_specific_si: p.r_specific_si };
        SolveSetup {
            solid: Arc::new(rasterize_wall(wall, &grid)),
            grid,
            gas,
            chamber: Chamber { p0: 1.0, t0: 1.0 },
            ambient: ambient_nd(p),
            numerics: Numerics::default(),
            refs: RefScales::from_chamber(p.r_throat_m, p.p0_pa, p.t0_k, &gas),
        }
    }

    /// Steps to VISUAL steady: the §9 criterion (supersonic core, barrel
    /// shock and Mach disk settled, ~5 plume transits) measured at ~6,100
    /// steps for the compact demo domain, scaled by domain length. The
    /// residual-based "settled" flag is NOT usable as a finish line here: the
    /// mildly overexpanded sea-level plume stays unsteady (shock-cell
    /// breathing) and the L2 residual plateaus above 1e-3 indefinitely, on
    /// the uniform historic grid and the graded one alike.
    fn steps_to_visual_steady(g: &Grid) -> u64 {
        (6100.0 * g.lz() / LZ).round() as u64
    }

    /// Runs to visual steady (or projects from measured steps/s when
    /// `project` is set — the uniform Long case exists only to be priced).
    fn record_bench(setting: &str, cells: usize, sps: f64, secs: f64) {
        // Results get committed, not reported in chat (CLAUDE.md).
        cfd_results::record_benchmark("grading-bench", cfd_results::Benchmark {
            case: "demo nozzle (gamma 1.24, eps 8, sea level)".into(),
            setting: setting.into(),
            cells: cells as u64,
            steps_per_sec: sps,
            seconds_to_steady: secs,
        });
    }

    fn run_case(name: &str, setup: SolveSetup) {
        let g = setup.grid.clone();
        let target = steps_to_visual_steady(&g);
        let mut s = cfd_core::EulerSolver::new(setup).unwrap();
        for _ in 0..50 {
            s.step().unwrap();
        }
        let t0 = Instant::now();
        for _ in 0..250 {
            s.step().unwrap();
        }
        let sps = 250.0 / t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mut info = s.step().unwrap();
        let mut floors = info.floor_activations;
        let mut last_floor = if floors > 0 { info.step } else { 0 };
        while info.step < target {
            info = s.step().unwrap();
            if info.floor_activations > floors {
                floors = info.floor_activations;
                last_floor = info.step;
            }
        }
        let wall_s = 300.0 / sps + t1.elapsed().as_secs_f64();
        let rep = s.report();
        println!(
            "{name}: {} x {} = {} cells | {:.0} steps/s | visual steady \
             (step {target}) in {:.0} s wall | floors {floors} (last at step {last_floor}) | \
             mdot {:.2} kg/s, C_f {:.3}, exit M {:.3}",
            g.nz, g.nr, g.len(), sps, wall_s,
            rep.mass_flow_kg_s, rep.thrust_coefficient, rep.exit_mach
        );
        record_bench(name, g.len(), sps, wall_s);
        cfd_results::record_note("grading-bench", &format!("{name}-report"), &format!(
            "{name} at visual steady (step {target}): mass flow {:.3} kg/s, C_f {:.4}, \
             exit Mach {:.3} (area-avg), p_e/p_a {:.3}, N_throat {:.0}.",
            rep.mass_flow_kg_s, rep.thrust_coefficient, rep.exit_mach,
            rep.exit_pressure_ratio, rep.cells_per_throat_radius));
        if floors > 0 {
            cfd_results::record_note("grading-bench", name, &format!(
                "{name}: {floors} positivity-floor activations, all during the cold-start \
                 front (last at step {last_floor}, zero after) — the §13 quarantine class."));
        }
        // Floor contact must have the §13 cold-start shape: confined to the
        // startup front and quiet ever after (the report quarantine covers
        // it). Steady-state floor contact is a hard failure. The startup
        // blast expands deeper before relief in the larger domains, so the
        // cold-start window is a few hundred steps at the lip corner.
        assert!(
            floors == 0 || last_floor < 1500,
            "{name}: floors {floors}, last at step {last_floor} — not startup-confined"
        );
    }

    fn preset_case(preset: DomainPreset) -> CaseParams {
        let (lz_rt, lr_rt, cells_per_rt, dz_over_dr) = preset.values();
        CaseParams { lz_rt, lr_rt, cells_per_rt, dz_over_dr, ..CaseParams::default() }
    }

    /// The work-order report (item 8): cells, steps/s, seconds-to-steady and
    /// the report numbers at each domain preset plus the old-default
    /// (graded-Standard tier) equivalent, on this machine.
    ///
    ///     cargo test -p cfd-ui grading_bench -- --include-ignored --nocapture
    #[test]
    #[ignore = "benchmark: tens of minutes of solver time; run explicitly"]
    fn grading_before_after() {
        let wall = conical_contour(8.0);

        // Reference: the historic compact uniform 320 x 200 demo grid.
        let pc = preset_case(DomainPreset::Preview);
        run_case("uniform Compact (historic)", bench_setup(base_grid(&pc), &pc, &wall));

        // The old default: the deleted Standard tier's extents through the
        // new direct fields (the grid is bit-identical to the old pipeline —
        // old_default_settings_reproduce_the_old_grid).
        let r_e = 8.0f64.sqrt();
        let po = CaseParams {
            lz_rt: LZ + 8.0 * r_e,
            lr_rt: LR * 1.3,
            ..CaseParams::default()
        };
        run_case("old Standard (equivalence)", make_setup(&po, &wall));

        // The three shortcuts; Large is the default.
        run_case("Preview preset", make_setup(&preset_case(DomainPreset::Preview), &wall));
        run_case("Standard preset", make_setup(&preset_case(DomainPreset::Standard), &wall));
        run_case("Large preset (default)", make_setup(&preset_case(DomainPreset::Large), &wall));
        cfd_results::record_note("grading-bench", "criterion",
            "seconds_to_steady uses the §9 VISUAL-steady criterion (~5 plume transits, \
             scaled by domain length). The residual-based settled flag does not fire at \
             sea level on either the uniform or the graded grid — the mildly overexpanded \
             plume keeps breathing above the 1e-3 threshold.");
    }
}

/// The payoff measurement for the contour swap: RS-25, 15° cone (what the app
/// drew before) vs the published 80% bell (what it draws now), at sea level
/// and at 20 km.
#[cfg(test)]
mod contour_before_after {
    use super::*;
    use cfd_contract::{Confidence, Solver};

    /// Total steps per run. Long enough that the last half is many domain
    /// flush times past the startup front.
    const TOTAL_STEPS: u64 = 12_000;
    /// THE DECLARED SETTLED WINDOW: the second half of every run, by rule,
    /// not by hand-picking. Everything asserted here is a mean over it.
    const WINDOW_START: u64 = TOTAL_STEPS / 2;
    /// Report sampling cadence inside the whole run.
    const SAMPLE_EVERY: u64 = 25;

    #[derive(Clone, Copy)]
    struct Sample {
        step: u64,
        time: f64,
        residual: f64,
        cf: f64,
        mdot: f64,
    }

    struct Trace {
        samples: Vec<Sample>,
        /// The last StepInfo — residual and `converged` as the solver reports
        /// them, not as the test wishes them to be.
        info: cfd_contract::StepInfo,
        confidence: Confidence,
        cells: (usize, usize),
    }

    impl Trace {
        fn window(&self) -> &[Sample] {
            let k = self
                .samples
                .partition_point(|s| s.step < WINDOW_START);
            &self.samples[k..]
        }
        /// Mean of `f` over a fraction of the settled window: (0.0, 0.5) is
        /// its first half, (0.5, 1.0) its second, (0.0, 1.0) the whole.
        fn mean(&self, lo: f64, hi: f64, f: impl Fn(&Sample) -> f64) -> f64 {
            let w = self.window();
            let (a, b) = ((w.len() as f64 * lo) as usize, (w.len() as f64 * hi) as usize);
            // f64 accumulation, as every reduction in this build must be.
            let acc: f64 = w[a..b].iter().map(&f).sum();
            acc / (b - a) as f64
        }
        fn span(&self, f: impl Fn(&Sample) -> f64 + Copy) -> (f64, f64) {
            self.window().iter().fold((f64::MAX, f64::MIN), |(lo, hi), s| {
                (lo.min(f(s)), hi.max(f(s)))
            })
        }
        fn window_time(&self) -> (f64, f64) {
            let w = self.window();
            (w[0].time, w[w.len() - 1].time)
        }
    }

    /// Run one case for `TOTAL_STEPS`, sampling the report on the way. The
    /// solver's own signals (`StepInfo::residual`, `.converged`, `.time`) are
    /// carried out with the samples: what "settled" means here is measured,
    /// not assumed.
    fn run_traced(label: &str, c: &CaseParams, wall: &[[f64; 2]]) -> Trace {
        let setup = make_setup(c, wall);
        let (nz, nr) = (setup.grid.nz, setup.grid.nr);
        let mut s = cfd_core::EulerSolver::new(setup).unwrap();
        let mut samples = Vec::with_capacity((TOTAL_STEPS / SAMPLE_EVERY) as usize + 1);
        let mut info = s.step().unwrap();
        while info.step < TOTAL_STEPS {
            info = s.step().unwrap();
            if info.step % SAMPLE_EVERY == 0 {
                let r = s.report();
                samples.push(Sample {
                    step: info.step,
                    time: info.time,
                    residual: info.residual,
                    cf: r.thrust_coefficient,
                    mdot: r.mass_flow_kg_s,
                });
            }
            if info.step % 2000 == 0 {
                println!(
                    "  {label}: step {} | t {:.2} | residual {:.3e} | converged {}",
                    info.step, info.time, info.residual, info.converged
                );
            }
        }
        let rep = s.report();
        Trace {
            samples,
            info,
            confidence: rep.confidence,
            cells: (nz, nr),
        }
    }

    /// Cone-vs-bell report comparison, ~25 min of solver time; run explicitly:
    ///
    ///     cargo test -p cfd-ui rs25_before_after -- --include-ignored --nocapture
    ///
    /// Two operating points, two roles:
    ///
    /// **Sea level** is the tasked report, and it is RECORDED, NOT ASSERTED.
    /// RS-25 at sea level is overexpanded (p_e/p_a ≈ 0.21 at the preset's
    /// γ = 1.20, under the 0.40 separation threshold) and a 30k-step probe
    /// found no steady state for either contour: the shock system breathes in
    /// and out of the divergent section (the §9 "shock inside the nozzle moves
    /// slowly" exception, taken to its limit) and every exit-plane readout
    /// oscillates with O(1) amplitude. "C_f rises with the bell" rests on the
    /// divergence-factor argument, which assumes attached full flow — at sea
    /// level it is untestable in this model, which is precisely what the §13
    /// separation warning ("readouts are not valid in this regime") says.
    ///
    /// **20 km** (p_a ≈ 5.5 kPa, p_e/p_a ≈ 3.9 — attached, strongly
    /// underexpanded) is where the claim is testable, and asserted: the cone's
    /// 15° exit loses ~1.7% of axial momentum to divergence ((1+cos 15°)/2)
    /// against the bell's ~7.3° exit (~0.4%), so C_f must rise.
    ///
    /// **What is asserted, and on what.** Neither contour CONVERGES: at 12k
    /// steps both sit well above `RESIDUAL_CONVERGED` (1e-3) and every
    /// `Report` carries `Confidence::NotConverged` — recorded, in the results
    /// file, in those words. So nothing here is asserted on an instant. Every
    /// asserted number is a mean over the DECLARED SETTLED WINDOW (the second
    /// half of the run, steps 6k–12k), and the window is qualified by a
    /// time-based steadiness check before the physics claim is read out of it:
    /// the drift of the window mean between its own halves must be small
    /// against the effect being claimed, and the residual must not be growing.
    /// A single-instant read is exactly what this test used to do, and it
    /// flipped sign before ~step 2000.
    ///
    /// COMPACT extents for all runs: these are exit-plane quantities, and the
    /// plume tail past the lip does not move them — it only costs solver time.
    /// The configurable-domain work order folded the plume extension into
    /// `preset_domain`, so the compact sizing (§8: 3(√ε−1) + 20 long, 2.5√ε
    /// high) is set explicitly here rather than selected by a tier.
    /// Model-to-model comparison, not a pad prediction.
    #[test]
    #[ignore = "solver benchmark: ~25 min; run explicitly with --include-ignored"]
    fn rs25_before_after() {
        let pre = PRESETS.iter().find(|p| p.name == "RS-25").unwrap();
        let mut c = pre.case(0.0, false);
        let s = c.area_ratio.sqrt();
        c.lz_rt = 3.0 * (s - 1.0) + 20.0;
        c.lr_rt = (2.5 * s).max(10.0);
        let walls = [
            ("15 deg cone (before)", conical_contour(c.area_ratio)),
            ("80% bell (after)", nozzle_contour(&c).points),
        ];

        // Sea level: record the tasked before/after, as window statistics of a
        // limit cycle rather than a single instant — there is no steady state
        // to snapshot.
        for (setting, wall) in &walls {
            let t = run_traced(setting, &c, wall);
            let (t0, t1) = t.window_time();
            let (cf_lo, cf_hi) = t.span(|s| s.cf);
            let (m_lo, m_hi) = t.span(|s| s.mdot);
            cfd_results::record_note("contour", &format!("rs25-sea-level-{setting}"), &format!(
                "RS-25 sea level, {setting}: over the settled window (steps {WINDOW_START}-{TOTAL_STEPS}, \
                 nd time {t0:.2}-{t1:.2}) mass flow {:.1} kg/s mean (range {m_lo:.1}-{m_hi:.1}), \
                 C_f {:.4} mean (range {cf_lo:.4}-{cf_hi:.4}), residual {:.2e} at the end, \
                 converged {}, confidence {:?} ({} x {} cells, Compact plume, floors {}). \
                 THE WINDOW IS A LIMIT CYCLE, not a steady state — see the \
                 rs25-sea-level-unsteady note.",
                t.mean(0.0, 1.0, |s| s.mdot), t.mean(0.0, 1.0, |s| s.cf),
                t.info.residual, t.info.converged, t.confidence,
                t.cells.0, t.cells.1, t.info.floor_activations));
            println!(
                "sea level, {setting}: mdot {:.1} kg/s | C_f {:.4} (range {cf_lo:.4}-{cf_hi:.4}) \
                 | residual {:.2e} | {:?}",
                t.mean(0.0, 1.0, |s| s.mdot), t.mean(0.0, 1.0, |s| s.cf),
                t.info.residual, t.confidence
            );
        }
        cfd_results::record_note("contour", "rs25-sea-level-unsteady",
            "RS-25 at sea level is overexpanded (p_e/p_a ~ 0.21 at the preset's gamma \
             1.20 — quasi-1D p_e ~ 21.6 kPa against 101.3 kPa ambient, below the 0.40 \
             separation threshold, the SS13 separation-warning regime) and has NO steady \
             state in this inviscid model: a 30k-step probe of both contours shows the \
             shock system breathing in and out of the nozzle, with exit-plane readouts \
             oscillating over mdot 291-738 kg/s (cone) / 48-895 (bell), exit Mach \
             0.58-2.97 / 0.77-3.59, C_f 1.41-1.86 / 1.43-1.79. The cone-vs-bell C_f \
             comparison is therefore asserted at 20 km (attached flow), not at sea level.");

        // 20 km: attached flow — the assertable comparison.
        c.altitude_m = 20_000.0;
        let mut tr = Vec::new();
        for (setting, wall) in &walls {
            let t = run_traced(setting, &c, wall);
            let (t0, t1) = t.window_time();
            let (cf_lo, cf_hi) = t.span(|s| s.cf);
            cfd_results::record_note("contour", &format!("rs25-20km-{setting}"), &format!(
                "RS-25 at 20 km, {setting}: over the settled window (steps {WINDOW_START}-{TOTAL_STEPS}, \
                 nd time {t0:.2}-{t1:.2}) mass flow {:.1} kg/s mean, C_f {:.4} mean \
                 (range {cf_lo:.4}-{cf_hi:.4}, half-window drift {:+.4}), residual {:.2e} \
                 at the end ({:.0}x RESIDUAL_CONVERGED = 1e-3), converged {}, confidence \
                 {:?} ({} x {} cells, Compact plume, floors {}).",
                t.mean(0.0, 1.0, |s| s.mdot), t.mean(0.0, 1.0, |s| s.cf),
                t.mean(0.5, 1.0, |s| s.cf) - t.mean(0.0, 0.5, |s| s.cf),
                t.info.residual, t.info.residual / 1e-3, t.info.converged, t.confidence,
                t.cells.0, t.cells.1, t.info.floor_activations));
            println!(
                "20 km, {setting}: mdot {:.1} kg/s | C_f {:.4} (range {cf_lo:.4}-{cf_hi:.4}) \
                 | residual {:.2e} | converged {} | {:?}",
                t.mean(0.0, 1.0, |s| s.mdot), t.mean(0.0, 1.0, |s| s.cf),
                t.info.residual, t.info.converged, t.confidence
            );
            tr.push(t);
        }
        let (cone, bell) = (&tr[0], &tr[1]);

        // ---- 1. Is the window steady enough to carry a mean? A REAL
        // time-based check, from the solver's own signals: over the declared
        // window the residual must not be growing, and the drift of the C_f
        // mean between the window's two halves must be small against the
        // effect being claimed. (The old "steadiness guard" compared two runs
        // at ONE instant, which tests a choked-flow identity, not time.)
        let d_cf = bell.mean(0.0, 1.0, |s| s.cf) - cone.mean(0.0, 1.0, |s| s.cf);
        let drift = |t: &Trace| t.mean(0.5, 1.0, |s| s.cf) - t.mean(0.0, 0.5, |s| s.cf);
        let drift_ratio = drift(cone).abs().max(drift(bell).abs()) / d_cf.abs();
        let res_growth = |t: &Trace| {
            t.mean(0.5, 1.0, |s| s.residual) / t.mean(0.0, 0.5, |s| s.residual)
        };
        let worst_growth = res_growth(cone).max(res_growth(bell));
        cfd_results::record_test("contour", cfd_results::TestResult {
            id: "rs25-cf-window-steady-20km".into(),
            name: format!(
                "RS-25 20 km settled window (steps {WINDOW_START}-{TOTAL_STEPS}): C_f half-window \
                 drift vs the claimed cone-to-bell rise"
            ).as_str().into(),
            expected: "<= 0.5".into(),
            actual: drift_ratio.into(),
            units: "|drift| / |delta C_f|".into(),
            pass: drift_ratio <= 0.5,
        });
        cfd_results::record_test("contour", cfd_results::TestResult {
            id: "rs25-residual-not-growing-20km".into(),
            name: "RS-25 20 km settled window: residual second half / first half".into(),
            expected: "<= 1.2".into(),
            actual: worst_growth.into(),
            units: "ratio".into(),
            pass: worst_growth <= 1.2,
        });
        cfd_results::record_note("contour", "rs25-20km-not-converged", &format!(
            "NEITHER 20 km run converges, and nothing here is asserted as if it did: at \
             step {TOTAL_STEPS} the residual is {:.2e} (cone) / {:.2e} (bell) against \
             RESIDUAL_CONVERGED = 1e-3 — {:.0}x / {:.0}x above it — StepInfo.converged is \
             false for both and every Report carries Confidence::{:?}. What IS asserted \
             is the mean over the declared settled window (steps {WINDOW_START}-{TOTAL_STEPS}, \
             nd time {:.2}-{:.2}), qualified first by a time-based steadiness check on \
             the same window: C_f half-window drift {:.4} / {:.4} (cone/bell) against a \
             cone-to-bell difference of {d_cf:.4}, and residual second-half/first-half \
             {:.2} / {:.2}. The single-instant version of this assert read +0.0218 at \
             step 5511 and NEGATIVE before step ~2000 — it was a snapshot of a wobble.",
            cone.info.residual, bell.info.residual,
            cone.info.residual / 1e-3, bell.info.residual / 1e-3, cone.confidence,
            cone.window_time().0, cone.window_time().1,
            drift(cone), drift(bell), res_growth(cone), res_growth(bell)));

        // ---- 2. Same throat, same chamber, same gas: the exit-plane mass
        // flow must agree. A choked-flow identity, NOT a steadiness check —
        // it holds at any instant of any choked flow, which is why it could
        // never have guarded steadiness. On window means, not an instant.
        let (m_cone, m_bell) = (cone.mean(0.0, 1.0, |s| s.mdot), bell.mean(0.0, 1.0, |s| s.mdot));
        let mdot_mismatch = (m_bell / m_cone - 1.0).abs();
        cfd_results::record_test("contour", cfd_results::TestResult {
            id: "rs25-mdot-agree-20km".into(),
            name: "RS-25 20 km mass flow, bell vs cone (same throat: choked-flow identity, \
                   NOT a steadiness check)".into(),
            expected: "<= 5% mismatch".into(),
            actual: mdot_mismatch.into(),
            units: "relative mismatch of window means".into(),
            pass: mdot_mismatch <= 0.05,
        });

        // ---- 3. The physics claim, on the settled-window mean.
        cfd_results::record_test("contour", cfd_results::TestResult {
            id: "rs25-cf-rise-20km".into(),
            name: format!(
                "RS-25 20 km C_f, 80% bell vs 15 deg cone (mean over the declared settled \
                 window, steps {WINDOW_START}-{TOTAL_STEPS})"
            ).as_str().into(),
            expected: "bell > cone".into(),
            actual: d_cf.into(),
            units: "delta C_f (window means)".into(),
            pass: d_cf > 0.0,
        });

        assert!(
            drift_ratio <= 0.5,
            "the settled window is not settled: C_f half-window drift is {drift_ratio:.2} of \
             the claimed rise (cone {:+.4}, bell {:+.4}, delta {d_cf:+.4}) — the window mean \
             cannot carry the claim",
            drift(cone), drift(bell)
        );
        assert!(
            worst_growth <= 1.2,
            "residual is growing across the settled window (worst second/first half ratio \
             {worst_growth:.2})"
        );
        assert!(
            mdot_mismatch <= 0.05,
            "same-throat mass flow disagrees at 20 km: cone {m_cone} vs bell {m_bell} kg/s"
        );
        assert!(
            d_cf > 0.0,
            "C_f did not rise with the bell at 20 km: cone {:.4} vs bell {:.4} (window means)",
            cone.mean(0.0, 1.0, |s| s.cf),
            bell.mean(0.0, 1.0, |s| s.cf)
        );
    }
}

#[cfg(test)]
mod floor_diag {
    use super::*;
    use cfd_contract::{FieldKind, Solver};

    #[test]
    #[ignore = "diagnostic"]
    fn diag_graded_large_floors() {
        let pl = CaseParams::default(); // the Large domain
        let wall = conical_contour(pl.area_ratio);
        let setup = make_setup(&pl, &wall);
        let g = setup.grid.clone();
        let mut s = cfd_core::EulerSolver::new(setup).unwrap();
        let mut prev = 0u64;
        let mut prints = 0;
        for _ in 0..17_000u64 {
            let info = s.step().unwrap();
            if info.floor_activations > prev {
                prev = info.floor_activations;
                if prints < 40 {
                    let snap = s.snapshot();
                    let (mut pmin, mut at) = (f64::INFINITY, (0usize, 0usize));
                    for ir in 0..g.nr {
                        for iz in 0..g.nz {
                            if snap.solid.is_solid(g.idx(iz, ir)) { continue; }
                            let p = snap.sample(FieldKind::Pressure, iz, ir) as f64;
                            if p < pmin { pmin = p; at = (iz, ir); }
                        }
                    }
                    println!(
                        "step {}: floors {} | min p {:.3e} nd at ({}, {}) z {:.1} r {:.2} dz {:.2}",
                        info.step, info.floor_activations, pmin / 5.0e6,
                        at.0, at.1, g.z_center(at.0), g.r_center(at.1), g.dz(at.0)
                    );
                    prints += 1;
                }
            }
        }
        println!("total floors {prev}");
    }
}

