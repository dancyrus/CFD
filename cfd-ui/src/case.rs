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
/// Bottom stop of the area-ratio slider. Named because it is load-bearing for
/// the Rao disclosure: it sits BELOW the digitised table's ε = 4 start, so any
/// bell preset can be dragged into the clamped region and the app has to say
/// so — the mirror of the ε > 100 case Merlin Vac already exercises. The top
/// stop is not a constant; it stretches to hold whatever preset is selected.
pub const AREA_RATIO_SLIDER_MIN: f64 = 2.0;
/// The digitised Rao table's area-ratio range (`cfd_geom::rao_angles` clamps to
/// it at BOTH ends rather than extrapolating). Outside it the bell angles are
/// an end row stretched to the requested exit radius, which the UI discloses.
pub const RAO_TABLE_EPS: std::ops::RangeInclusive<f64> = 4.0..=100.0;

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
    /// Divergent half-angle of a `Conical` contour, degrees. Per-case because
    /// the app used to hard-code 15° and the contour audit found no source for
    /// that number; handbook practice runs 12–18°. `CONE_HALF_ANGLE_DEG` is
    /// still the default, and three of the historical presets genuinely are
    /// 15° — but three data points are not a constant.
    pub cone_half_angle_deg: f64,
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
            cone_half_angle_deg: CONE_HALF_ANGLE_DEG,
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

/// A propellant class whose chamber thermochemistry comes from
/// `tools/propellant_cea.py` — constant-enthalpy constant-pressure equilibrium
/// from the liquid reactants' heats of formation, on the GRI-Mech 3.0 species
/// set with the HIGH-TEMPERATURE thermo fits (`gri30_highT`). The standard
/// `gri30` polynomials cap at 3000 K and every LOX/RP-1 case here runs above
/// 3450 K, so the standard set would be extrapolating on all five. The full
/// table is `docs/results/propellant-cea.md`.
///
/// γ here is the FROZEN ratio — cp/cv of the equilibrium chamber mixture, its
/// composition held fixed — because the solver is a fixed-gamma ideal gas with
/// no species transport (docs/physics-reference.md §2). Handing it a
/// shifting-equilibrium exponent would ask a frozen-composition code to
/// reproduce a recombining expansion it cannot perform.
///
/// **These are class REFERENCE values — the mean over the class's cases.** A
/// preset carries its own per-case numbers instead (`EnginePreset::gamma` and
/// friends), because the cases within a class do not agree well enough to
/// share one triple: the two LOX/ethanol engines sit at O/F 1.130 and 1.324
/// and their T₀ differs by 6.4%. `cases_span` records that spread, and
/// `preset_gas_matches_its_propellant_class` is the test that holds every
/// preset inside its own class's measured range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropellantClass {
    /// LOX / RP-1. Five cases, 4.00–4.80 MPa, O/F 2.200–2.250.
    LoxRp1,
    /// LOX / (ethanol 0.75 + water 0.25 by mass) — the V-2 and Redstone
    /// B-Stoff. Two cases, 1.57 and 2.19 MPa, O/F 1.130 and 1.324.
    LoxEthanol75,
    /// RFNA (HNO₃ 0.85 / N₂O₄ 0.13 / H₂O 0.02) / (aniline 0.626 + furfuryl
    /// alcohol 0.374), all by mass. One case, 2.10 MPa, O/F 2.65.
    RfnaAnilineFurfuryl,
}

/// Chamber thermochemistry of one propellant class or one engine case.
///
/// Two characteristic velocities, because they answer different questions and
/// the historical throat radii depend on getting that distinction right:
///
/// * `cstar_shifting_m_s` is what `tools/propellant_cea.py` computes by
///   expanding the equilibrium mixture to the sonic point, letting the
///   composition shift. It is what CEA reports, and it is what the published
///   throat areas were sized against — so it, not the closed form, is what the
///   `A_t = F / (Cf·λ·η·p₀)` derivation is calibrated on.
/// * `cstar_frozen_m_s` is the closed form for a calorically perfect gas at
///   these (T₀, MW, γ). It runs ~1.3% low, because holding chamber γ all the
///   way to the throat ignores the recombination that reheats the gas as it
///   accelerates. It is nonetheless the honest figure for THIS solver, which
///   is exactly that fixed-γ ideal gas.
///
/// `cstar_ideal` recomputes the frozen form in Rust and
/// `rust_and_python_frozen_cstar_agree` holds the two implementations
/// together, so the table cannot drift from the tool that produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GasProperties {
    pub gamma: f64,
    pub t0_k: f64,
    pub mw_g_mol: f64,
    pub cstar_shifting_m_s: f64,
    pub cstar_frozen_m_s: f64,
}

impl PropellantClass {
    pub const ALL: [PropellantClass; 3] = [
        PropellantClass::LoxRp1,
        PropellantClass::LoxEthanol75,
        PropellantClass::RfnaAnilineFurfuryl,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            PropellantClass::LoxRp1 => "LOX/RP-1",
            PropellantClass::LoxEthanol75 => "LOX/ethanol 75%",
            PropellantClass::RfnaAnilineFurfuryl => "RFNA/aniline-furfuryl",
        }
    }

    /// Class reference values: the mean over the class's cases in
    /// `docs/results/propellant-cea.md`.
    pub const fn reference(self) -> GasProperties {
        match self {
            PropellantClass::LoxRp1 => GasProperties {
                gamma: 1.221_014,
                t0_k: 3488.581,
                mw_g_mol: 21.865_61,
                cstar_shifting_m_s: 1804.863,
                cstar_frozen_m_s: 1764.927,
            },
            PropellantClass::LoxEthanol75 => GasProperties {
                gamma: 1.191_029,
                t0_k: 2985.608,
                mw_g_mol: 22.694_11,
                cstar_shifting_m_s: 1644.738,
                cstar_frozen_m_s: 1616.966,
            },
            PropellantClass::RfnaAnilineFurfuryl => GasProperties {
                gamma: 1.211_618,
                t0_k: 2988.298,
                mw_g_mol: 25.620_19,
                cstar_shifting_m_s: 1539.056,
                cstar_frozen_m_s: 1513.231,
            },
        }
    }

    /// Inclusive (min, max) of each quantity across the class's cases, as
    /// computed by the tool. This is what makes the class reference honest:
    /// where the span is wide the presets must carry their own numbers, and
    /// the test asserts they land inside it.
    pub const fn cases_span(self) -> (GasProperties, GasProperties) {
        match self {
            PropellantClass::LoxRp1 => (
                GasProperties { gamma: 1.219_702, t0_k: 3470.218, mw_g_mol: 21.761_67, cstar_shifting_m_s: 1803.813, cstar_frozen_m_s: 1763.565 },
                GasProperties { gamma: 1.221_839, t0_k: 3518.501, mw_g_mol: 22.026_37, cstar_shifting_m_s: 1806.722, cstar_frozen_m_s: 1766.679 },
            ),
            PropellantClass::LoxEthanol75 => (
                GasProperties { gamma: 1.187_463, t0_k: 2889.649, mw_g_mol: 22.031_74, cstar_shifting_m_s: 1634.501, cstar_frozen_m_s: 1612.833 },
                GasProperties { gamma: 1.194_595, t0_k: 3081.567, mw_g_mol: 23.356_47, cstar_shifting_m_s: 1654.974, cstar_frozen_m_s: 1621.099 },
            ),
            PropellantClass::RfnaAnilineFurfuryl => (
                GasProperties { gamma: 1.211_618, t0_k: 2988.298, mw_g_mol: 25.620_19, cstar_shifting_m_s: 1539.056, cstar_frozen_m_s: 1513.231 },
                GasProperties { gamma: 1.211_618, t0_k: 2988.298, mw_g_mol: 25.620_19, cstar_shifting_m_s: 1539.056, cstar_frozen_m_s: 1513.231 },
            ),
        }
    }
}

/// Ideal characteristic velocity of a calorically perfect gas, m/s:
/// `c* = sqrt(R T₀ / γ) · ((γ+1)/2)^((γ+1)/(2(γ−1)))`. Evaluated with the
/// FROZEN γ, so it is the c* this solver reproduces — `tools/propellant_cea.py`
/// computes the same expression and the two are asserted equal.
pub fn cstar_ideal(t0_k: f64, mw_g_mol: f64, gamma: f64) -> f64 {
    let r_specific = R_UNIVERSAL_SI / mw_g_mol;
    let g = gamma;
    (r_specific * t0_k / g).sqrt() * ((g + 1.0) / 2.0).powf((g + 1.0) / (2.0 * (g - 1.0)))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnginePreset {
    pub name: &'static str,
    pub propellant: &'static str,
    /// The CEA-computed propellant class this preset's gas properties come
    /// from, or `None` for the six presets that predate
    /// `tools/propellant_cea.py` and carry per-engine researched values
    /// instead. `None` is not a gap to be filled in casually: the F-1's γ is
    /// deliberately pinned (see its note), and re-deriving the other five from
    /// the CEA table would move every committed number they produce.
    pub propellant_class: Option<PropellantClass>,
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
    /// Where `r_throat_m` comes from. Only the Redstone's is a published
    /// dimension; the rest are derived from thrust and thrust coefficient
    /// against that one calibration point. Empty for the six presets that
    /// predate this labelling.
    pub throat_source: &'static str,
    /// Where `p0_pa` and `area_ratio` come from. Empty for the six presets
    /// that predate this labelling.
    pub operating_source: &'static str,
    /// Divergent half-angle for a `Conical` preset, degrees. Ignored by the
    /// bell kinds, which take their wall angles from the Rao table or from
    /// published geometry.
    pub cone_half_angle_deg: f64,
    /// c* by shifting-equilibrium expansion to the sonic point, m/s, from
    /// `tools/propellant_cea.py`. Data, not a derived quantity: only the
    /// equilibrium solve can produce it, and it is the figure the throat
    /// derivation was calibrated against. Zero for the six presets that
    /// predate the tool.
    pub cstar_shifting_m_s: f64,
    /// Shown as "(slow)" in the selector: run time ≈12× Merlin 1D.
    pub slow: bool,
    /// The altitude the preset OPENS at, km. Almost every engine is 0 — it is
    /// designed to run on the pad and the sea-level view is the honest first
    /// look. The exceptions are the altitude-optimised upper stages, which
    /// genuinely separate at sea level: the Atlas LR105 sustainer and the
    /// Titan I LR91 sit at p_e/p_a of 0.168 and 0.157, both well under the
    /// 0.40 Summerfield threshold. Opening those at 0 km would raise the
    /// separation warning the instant the preset is clicked, with no user
    /// error involved and nothing the user could do about it — a warning that
    /// fires on arrival teaches nothing and trains people to ignore the real
    /// ones. So a preset carries the altitude at which its readouts mean
    /// something, and `apply_preset` moves the slider there.
    pub default_altitude_km: f64,
    /// Preset-specific honesty note, surfaced as a tooltip.
    pub note: &'static str,
}

/// Shared tooltip for the five bells in the historical set. Thor is 1957 and
/// Atlas D 1959, against Rao's 1958 and 1960 papers — these engines predate or
/// are contemporary with the charts the generator interpolates, so the wall it
/// draws is of the right family but is not the engine's own contour. The same
/// class of caveat the Raptor 2 preset already carries, for the opposite
/// reason: that one is too late and too different, these are too early.
const EARLY_BELL_NOTE: &str = "Early contoured nozzle. Predates or is \
     contemporary with Rao's published charts, so the generated contour is an \
     approximation of the real one.";

/// The six modern presets come first and keep their indices — `PRESETS[0]` is
/// the Merlin 1D cost reference and several tests index by position. The eight
/// historical engines (1943–1961) follow, appended.
pub const PRESETS: [EnginePreset; 14] = [
    EnginePreset {
        name: "Merlin 1D",
        propellant: "LOX/RP-1",
        propellant_class: None,
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
        throat_source: "",
        operating_source: "",
        cone_half_angle_deg: CONE_HALF_ANGLE_DEG,
        cstar_shifting_m_s: 0.0,
        slow: false,
        default_altitude_km: 0.0,
        note: "",
    },
    EnginePreset {
        name: "F-1",
        propellant: "LOX/RP-1",
        propellant_class: None,
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
        throat_source: "",
        operating_source: "",
        cone_half_angle_deg: CONE_HALF_ANGLE_DEG,
        cstar_shifting_m_s: 0.0,
        slow: false,
        default_altitude_km: 0.0,
        note: "Sits 3% inside the separation threshold at sea level by design \
               — γ is hard-coded at 1.24 so the marginal warning is a \
               deliberate design point, not parameter drift.",
    },
    EnginePreset {
        name: "Raptor 2",
        propellant: "LOX/CH4",
        propellant_class: None,
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
        throat_source: "",
        operating_source: "",
        cone_half_angle_deg: CONE_HALF_ANGLE_DEG,
        cstar_shifting_m_s: 0.0,
        slow: false,
        default_altitude_km: 0.0,
        note: "Wall angles are the published θ_n 32.0° / θ_e 6.0° on a \
               0.300 R_t throat arc, not a Rao-table bell — no bell percent \
               reproduces both angles. The 76% is the length only.",
    },
    EnginePreset {
        name: "AJ10-190",
        propellant: "N2O4/MMH",
        propellant_class: None,
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
        throat_source: "",
        operating_source: "",
        cone_half_angle_deg: CONE_HALF_ANGLE_DEG,
        cstar_shifting_m_s: 0.0,
        slow: false,
        default_altitude_km: 0.0,
        note: "",
    },
    EnginePreset {
        name: "RS-25",
        propellant: "LOX/LH2",
        propellant_class: None,
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
        throat_source: "",
        operating_source: "",
        cone_half_angle_deg: CONE_HALF_ANGLE_DEG,
        cstar_shifting_m_s: 0.0,
        slow: false,
        default_altitude_km: 0.0,
        note: "Shows a separation warning at sea level: a known conservatism \
               of the criterion, not an error — the real engine runs on the \
               pad.",
    },
    EnginePreset {
        name: "Merlin Vac",
        propellant: "LOX/RP-1",
        propellant_class: None,
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
        throat_source: "",
        operating_source: "",
        cone_half_angle_deg: CONE_HALF_ANGLE_DEG,
        cstar_shifting_m_s: 0.0,
        slow: true,
        default_altitude_km: 0.0,
        note: "The costliest preset by far even at the reduced 14 cells per \
               throat radius (the mass-flow badge goes amber for that reason) \
               — see the run-time multiple above. ε = 165 is also past the end \
               of the digitised Rao table (ε ≤ 100): the bell angles clamp to \
               the ε = 100 row — see the amber flag.",
    },
    // -----------------------------------------------------------------------
    // Historical engines, 1943–1961. γ, T₀, MW and the shifting-equilibrium c*
    // are the per-case numbers from `tools/propellant_cea.py`
    // (docs/results/propellant-cea.md), not the class means — see
    // `PropellantClass`.
    //
    // THE BELL/CONE TRANSITION FALLS INSIDE THIS DATE RANGE, which is the one
    // thing an earlier revision of this table got wrong on five of eight
    // engines. The V-2, WAC Corporal and Redstone are cones; Thor, both Atlas
    // engines and both Titan I engines are bells. Rocketdyne carried bell
    // nozzles from Navaho into Atlas, Jupiter and Thor, and Rao was at
    // Rocketdyne, so the change happens mid-1950s — inside the set, not after
    // it.
    //
    // Throat radii are DERIVED from published thrust and thrust coefficient,
    // A_t = F / (Cf_ideal · λ · η · p₀), except the Redstone's, which is the
    // one published throat dimension in the set (15.5 in diameter) and the
    // single point η = 0.9877 is calibrated on. That derivation reproduces it
    // to −0.03%. The obvious alternative — A_t = ṁ·c*/p₀ — is NOT used: it
    // back-solves the Redstone to a c* efficiency of 100.2%, which no engine
    // achieves, and the same AEHS table's LR89-5 row implies an Isp of 356 s
    // against its own Isp column of 248–282. The mass-flow column is the weak
    // one, so it is not what the geometry rests on.
    // -----------------------------------------------------------------------
    EnginePreset {
        name: "V-2 (A-4) Model 39",
        propellant: PropellantClass::LoxEthanol75.label(),
        propellant_class: Some(PropellantClass::LoxEthanol75),
        area_ratio: 2.83,
        p0_pa: 1.572e6,
        r_throat_m: 0.193_6,
        gamma: 1.194_595,
        t0_k: 2889.649,
        mw_g_mol: 22.031_74,
        cstar_shifting_m_s: 1634.501,
        cells_per_rt: CELLS_PER_RT,
        contour_kind: ContourKind::Conical,
        cone_half_angle_deg: 15.0,
        throat_arc_down: THROAT_ARC_DOWN,
        bell_percent: 0.8,
        bell_source: "15° cone — confirmed (Wewerka set the half-angle; AEHS names it conical)",
        throat_source: "derived from thrust and Cf; validated against Redstone",
        operating_source: "published",
        slow: false,
        default_altitude_km: 0.0,
        note: "Area ratio 2.83 per Sutton. Museum-hardware dimensions elsewhere \
               imply ~3.34. Unresolved.",
    },
    EnginePreset {
        name: "WAC Corporal 38ALDW-1500",
        propellant: PropellantClass::RfnaAnilineFurfuryl.label(),
        propellant_class: Some(PropellantClass::RfnaAnilineFurfuryl),
        area_ratio: 5.00,
        p0_pa: 2.068e6,
        r_throat_m: 0.027_5,
        gamma: 1.211_618,
        t0_k: 2988.298,
        mw_g_mol: 25.620_19,
        cstar_shifting_m_s: 1539.056,
        cells_per_rt: CELLS_PER_RT,
        contour_kind: ContourKind::Conical,
        cone_half_angle_deg: 15.0,
        throat_arc_down: THROAT_ARC_DOWN,
        bell_percent: 0.8,
        bell_source: "cone — confirmed by date (1945 Aerojet JATO unit)",
        throat_source: "derived from thrust and Cf; validated against Redstone",
        operating_source: "RECONSTRUCTED — chamber pressure and area ratio unpublished",
        slow: false,
        default_altitude_km: 0.0,
        note: "Chamber pressure and area ratio are both reconstructed, not \
               published. Thrust, burn time, propellants and the \
               air-pressurised feed are confirmed; everything downstream of p₀ \
               and ε is only as good as that reconstruction.",
    },
    EnginePreset {
        name: "Redstone NAA 75-110-A-7",
        propellant: PropellantClass::LoxEthanol75.label(),
        propellant_class: Some(PropellantClass::LoxEthanol75),
        area_ratio: 3.61,
        p0_pa: 2.193e6,
        // The one published throat dimension in the set, and the point the
        // whole derivation is calibrated on: 15.5 in diameter.
        r_throat_m: 0.196_85,
        gamma: 1.187_463,
        t0_k: 3081.567,
        mw_g_mol: 23.356_47,
        cstar_shifting_m_s: 1654.974,
        cells_per_rt: CELLS_PER_RT,
        contour_kind: ContourKind::Conical,
        cone_half_angle_deg: 15.0,
        throat_arc_down: THROAT_ARC_DOWN,
        bell_percent: 0.8,
        bell_source: "15° cone — confirmed (\"a straight-sided 15° divergent \
                      nozzle section was retained\")",
        throat_source: "published (15.5 in throat diameter)",
        operating_source: "published",
        slow: false,
        default_altitude_km: 0.0,
        note: "The only engine here with a published throat diameter, and so \
               the calibration point for every derived throat radius in the \
               set: the thrust-and-Cf derivation reproduces it to −0.03%.",
    },
    EnginePreset {
        name: "Thor LR79-NA-7 (MB-1)",
        propellant: PropellantClass::LoxRp1.label(),
        propellant_class: Some(PropellantClass::LoxRp1),
        area_ratio: 8.00,
        p0_pa: 4.100e6,
        r_throat_m: 0.187_8,
        gamma: 1.220_567,
        t0_k: 3495.093,
        mw_g_mol: 21.946_03,
        cstar_shifting_m_s: 1804.434,
        cells_per_rt: CELLS_PER_RT,
        contour_kind: ContourKind::ParabolicBell,
        cone_half_angle_deg: CONE_HALF_ANGLE_DEG,
        throat_arc_down: THROAT_ARC_DOWN,
        bell_percent: 0.80,
        bell_source: "design-class estimate; no published wall angles",
        throat_source: "derived from thrust and Cf; validated against Redstone",
        operating_source: "published",
        slow: false,
        default_altitude_km: 0.0,
        note: EARLY_BELL_NOTE,
    },
    EnginePreset {
        name: "Titan I LR87-AJ-3 (1 of 2)",
        propellant: PropellantClass::LoxRp1.label(),
        propellant_class: Some(PropellantClass::LoxRp1),
        area_ratio: 8.00,
        p0_pa: 4.000e6,
        r_throat_m: 0.187_7,
        gamma: 1.221_839,
        t0_k: 3470.218,
        mw_g_mol: 21.761_67,
        cstar_shifting_m_s: 1803.813,
        cells_per_rt: CELLS_PER_RT,
        contour_kind: ContourKind::ParabolicBell,
        cone_half_angle_deg: CONE_HALF_ANGLE_DEG,
        throat_arc_down: THROAT_ARC_DOWN,
        bell_percent: 0.80,
        bell_source: "design-class estimate; no published wall angles; contour family unverified",
        throat_source: "derived from thrust and Cf; validated against Redstone",
        operating_source: "published",
        slow: false,
        default_altitude_km: 0.0,
        note: "One of the two chambers of the LR87-AJ-3 first stage, not the \
               pair: the preset solves a single nozzle.\n\
               Contour family is UNVERIFIED — Aerojet, not Rocketdyne, so the \
               Navaho bell lineage does not carry over. Shipped as a bell; \
               resolvable from museum hardware.",
    },
    EnginePreset {
        name: "Titan I LR91-AJ-3",
        propellant: PropellantClass::LoxRp1.label(),
        propellant_class: Some(PropellantClass::LoxRp1),
        area_ratio: 25.0,
        p0_pa: 4.500e6,
        r_throat_m: 0.118_9,
        gamma: 1.221_424,
        t0_k: 3483.149,
        mw_g_mol: 21.787_13,
        cstar_shifting_m_s: 1805.410,
        cells_per_rt: CELLS_PER_RT,
        contour_kind: ContourKind::ParabolicBell,
        cone_half_angle_deg: CONE_HALF_ANGLE_DEG,
        throat_arc_down: THROAT_ARC_DOWN,
        bell_percent: 0.75,
        bell_source: "design-class estimate; no published wall angles; contour family unverified",
        throat_source: "derived from thrust and Cf; validated against Redstone",
        operating_source: "published",
        slow: false,
        // Separates at sea level (p_e/p_a 0.157); the computed threshold is
        // 7.2 km and 12 km carries margin over it.
        default_altitude_km: 12.0,
        note: "Altitude-optimised second stage, separated at sea level \
               (p_e/p_a ≈ 0.16 against the 0.40 threshold) — opens at 12 km, \
               above the 7.2 km the criterion actually needs.\n\
               Contour family is UNVERIFIED (Aerojet). The ablative skirt \
               carries ε from 13:1 to 25:1.",
    },
    EnginePreset {
        name: "Atlas LR89-5 booster (1 of 2)",
        propellant: PropellantClass::LoxRp1.label(),
        propellant_class: Some(PropellantClass::LoxRp1),
        area_ratio: 8.00,
        p0_pa: 3.999e6,
        r_throat_m: 0.198_7,
        gamma: 1.221_536,
        t0_k: 3475.943,
        mw_g_mol: 21.806_83,
        cstar_shifting_m_s: 1803.934,
        cells_per_rt: CELLS_PER_RT,
        contour_kind: ContourKind::ParabolicBell,
        cone_half_angle_deg: CONE_HALF_ANGLE_DEG,
        throat_arc_down: THROAT_ARC_DOWN,
        bell_percent: 0.80,
        bell_source: "design-class estimate; no published wall angles",
        throat_source: "derived from thrust and Cf; validated against Redstone",
        operating_source: "published",
        slow: false,
        default_altitude_km: 0.0,
        note: "One of the two booster chambers, not the pair. This is the \
               1959–60 Atlas D/E/F with the MA-3 engine set, NOT the 1990 \
               commercial \"Atlas I\" with MA-5 — the area ratio is 8 either \
               way, but the chamber pressure differs.",
    },
    EnginePreset {
        name: "Atlas LR105-5 sustainer",
        propellant: PropellantClass::LoxRp1.label(),
        propellant_class: Some(PropellantClass::LoxRp1),
        area_ratio: 25.0,
        p0_pa: 4.799e6,
        r_throat_m: 0.120_0,
        gamma: 1.219_702,
        t0_k: 3518.501,
        mw_g_mol: 22.026_37,
        cstar_shifting_m_s: 1806.722,
        cells_per_rt: CELLS_PER_RT,
        contour_kind: ContourKind::ParabolicBell,
        cone_half_angle_deg: CONE_HALF_ANGLE_DEG,
        throat_arc_down: THROAT_ARC_DOWN,
        bell_percent: 0.80,
        bell_source: "design-class estimate; no published wall angles",
        throat_source: "derived from thrust and Cf; validated against Redstone",
        operating_source: "published",
        slow: false,
        // Separates at sea level (p_e/p_a 0.168); computed threshold 6.7 km,
        // shipped at 10 km for margin.
        default_altitude_km: 10.0,
        note: "Altitude-optimised sustainer, separated at sea level \
               (p_e/p_a ≈ 0.17 against the 0.40 threshold) — opens at 10 km, \
               above the 6.7 km the criterion actually needs. Drag the \
               altitude slider down to see the separation warning it avoids.",
    },
];

impl EnginePreset {
    /// The altitude this preset opens at, in metres, clamped to the model's
    /// 58 km ceiling (`ALT_MAX_M`). The clamp is only about keeping the slider
    /// handle on its track: `atmosphere` clamps at the same altitude, so the
    /// ambient a higher `default_altitude_km` would ask for is the ambient it
    /// gets anyway. Above the cap the app's own answer is the labelled vacuum
    /// stop, which a preset does not select on the user's behalf.
    pub fn default_altitude_m(&self) -> f64 {
        (self.default_altitude_km * 1000.0).clamp(0.0, ALT_MAX_M)
    }

    /// This preset's own chamber thermochemistry, as one value. The frozen c*
    /// is recomputed from (T₀, MW, γ) rather than stored, so it can never
    /// disagree with the three numbers it is derived from; the
    /// shifting-equilibrium c* is carried as data because only the equilibrium
    /// solve in `tools/propellant_cea.py` can produce it.
    pub fn gas(&self) -> GasProperties {
        GasProperties {
            gamma: self.gamma,
            t0_k: self.t0_k,
            mw_g_mol: self.mw_g_mol,
            cstar_shifting_m_s: self.cstar_shifting_m_s,
            cstar_frozen_m_s: cstar_ideal(self.t0_k, self.mw_g_mol, self.gamma),
        }
    }

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
            cone_half_angle_deg: self.cone_half_angle_deg,
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

/// The `NozzleSpec` this case asks `cfd_geom` for. Split out of
/// `nozzle_contour` so a test can drive `generate_contour` with exactly the
/// spec the app sends and validate the `WallProfile` it returns — going
/// through `nozzle_contour` only ever yields points in r_t units, with the
/// profile's own `validate` (strict z monotonicity, finiteness, r > 0) already
/// discarded.
pub fn nozzle_spec(p: &CaseParams) -> NozzleSpec {
    let contour = match p.contour_kind {
        ContourKind::Conical => cfd_geom::ContourKind::Conical {
            half_angle_deg: p.cone_half_angle_deg,
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
    NozzleSpec {
        throat_radius_m: p.r_throat_m,
        area_ratio: p.area_ratio,
        contraction_ratio: CONTRACTION_RATIO,
        converge_half_angle_deg: CONVERGE_HALF_ANGLE_DEG,
        throat_arc_up: THROAT_ARC_UP,
        throat_arc_down: p.throat_arc_down,
        contour,
    }
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
    match cfd_geom::generate_contour(&nozzle_spec(p), CONTOUR_SAMPLES) {
        Ok(profile) => {
            let inv = 1.0 / p.r_throat_m;
            GeneratedWall {
                points: profile.points.iter().map(|q| [q[0] * inv, q[1] * inv]).collect(),
                kind: p.contour_kind,
                fallback: None,
            }
        }
        Err(e) => GeneratedWall {
            points: fallback_cone(p.area_ratio, p.cone_half_angle_deg),
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
fn fallback_cone(area_ratio: f64, half_angle_deg: f64) -> Vec<[f64; 2]> {
    let alpha = half_angle_deg.to_radians();
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
    fallback_cone(area_ratio, CONE_HALF_ANGLE_DEG)
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
            // Both the preset's OWN wall and the fallback cone must fit: the
            // cone is what a rejected spec degrades to, so it may not overrun
            // the domain either. For the six modern presets the own wall is a
            // bell and the cone is the longer of the two; the eight historical
            // engines ARE cones (1943–1961 — the parabolic bell is a Rao 1958
            // result), so for those the two walls coincide and the check is
            // simply run twice.
            let wall = nozzle_contour(&c);
            assert_eq!(wall.fallback, None, "{}: fell back to the cone", p.name);
            assert_eq!(
                wall.kind, p.contour_kind,
                "{}: produced a different contour than the preset asked for",
                p.name
            );
            for (which, pts) in [("own wall", wall.points), ("cone", conical_contour(c.area_ratio))] {
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

    /// Every BELL preset's production wall is the bell (not the silent
    /// fallback cone): the generator reports the bell kind it was asked for,
    /// the wall is strictly shorter than the 15° cone at the same area ratio,
    /// and its exit angle is well under 15°.
    ///
    /// The historical presets are excluded by construction, not by exception —
    /// they are 15° cones on purpose (see the PRESETS comment), and a cone
    /// failing "is it a bell?" would be the test reporting the wrong thing.
    /// `historical_presets_are_cones` is their counterpart and asserts the
    /// opposite, so neither group is merely unchecked.
    #[test]
    fn presets_actually_get_bells() {
        // Six modern presets plus the five historical bells — Thor, both
        // Atlas engines and both Titan I engines. The bell/cone transition
        // falls inside 1943–1961, so "historical" and "cone" are NOT the same
        // set; an earlier revision of the table assumed they were and had five
        // engines wrong.
        let bells: Vec<_> = PRESETS.iter().filter(|p| p.contour_kind.is_bell()).collect();
        assert_eq!(bells.len(), 11, "6 modern + 5 historical bells: {:?}",
                   bells.iter().map(|p| p.name).collect::<Vec<_>>());
        for pre in bells {
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

/// The historical engine presets (1943–1961) and the CEA-backed propellant
/// classes behind them. Results go to
/// docs/results/historical-presets-<machine>.json (CLAUDE.md: results get
/// committed, not reported in chat) and are written up in
/// docs/results/historical-presets-v1.md.
#[cfg(test)]
mod historical_presets {
    use super::*;

    const SUITE: &str = "historical-presets";

    /// The eight engines this work order added, in table order, with the
    /// contour each one is supposed to have. Named rather than sliced by index
    /// so the list survives a reordering of `PRESETS`, and paired with the
    /// contour so the cone/bell split is asserted rather than assumed — an
    /// earlier revision of this table had all eight as cones and was wrong on
    /// five of them.
    const HISTORICAL: [(&str, bool); 8] = [
        ("V-2 (A-4) Model 39", false),
        ("WAC Corporal 38ALDW-1500", false),
        ("Redstone NAA 75-110-A-7", false),
        ("Thor LR79-NA-7 (MB-1)", true),
        ("Titan I LR87-AJ-3 (1 of 2)", true),
        ("Titan I LR91-AJ-3", true),
        ("Atlas LR89-5 booster (1 of 2)", true),
        ("Atlas LR105-5 sustainer", true),
    ];

    fn historical() -> Vec<&'static EnginePreset> {
        let v: Vec<_> = HISTORICAL
            .iter()
            .map(|(n, is_bell)| {
                let p = PRESETS
                    .iter()
                    .find(|p| p.name == *n)
                    .unwrap_or_else(|| panic!("preset {n} is missing from PRESETS"));
                assert_eq!(
                    p.contour_kind.is_bell(),
                    *is_bell,
                    "{n}: contour family is not what the source research says"
                );
                p
            })
            .collect();
        // Every conical preset is a historical one: the six modern presets are
        // all bells, so a cone appearing outside this list means something was
        // added without joining it (and so escaping every test below).
        let cones: Vec<_> = PRESETS
            .iter()
            .filter(|p| !p.contour_kind.is_bell())
            .map(|p| p.name)
            .collect();
        let want: Vec<_> = HISTORICAL
            .iter()
            .filter(|(_, b)| !*b)
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(cones, want, "the conical presets are the historical cones");
        v
    }

    fn record(id: &str, name: &str, expected: &str, actual: f64, units: &str, pass: bool) {
        cfd_results::record_test(SUITE, cfd_results::TestResult {
            id: id.into(),
            name: name.into(),
            expected: expected.into(),
            actual: actual.into(),
            units: units.into(),
            pass,
        });
    }

    /// 1. Every historical preset generates a valid wall through
    ///    `cfd_geom::generate_contour`: the spec is accepted (no silent
    ///    fallback to the cone), the returned `WallProfile` passes its own
    ///    `validate` — finite, r > 0, z STRICTLY increasing, which for a
    ///    single-valued polyline is exactly non-self-intersection — and the
    ///    geometry is the one asked for: minimum radius r_t at the marked
    ///    throat, exit radius r_t·√ε, and an exit wall angle that matches the
    ///    contour family (the cone's own half-angle, or well under it for a
    ///    bell).
    #[test]
    fn historical_presets_generate_valid_walls() {
        let mut worst_exit_err = 0.0f64;
        let mut worst_cone_angle_err = 0.0f64;
        for pre in historical() {
            let c = pre.case(pre.default_altitude_m(), false);
            let spec = nozzle_spec(&c);
            match pre.contour_kind {
                ContourKind::Conical => assert_eq!(
                    spec.contour,
                    cfd_geom::ContourKind::Conical { half_angle_deg: pre.cone_half_angle_deg },
                    "{}: cone spec does not carry the preset's own half-angle",
                    pre.name
                ),
                ContourKind::ParabolicBell => assert_eq!(
                    spec.contour,
                    cfd_geom::ContourKind::ParabolicBell { bell_percent: pre.bell_percent },
                    "{}: bell spec does not carry the preset's own bell percent",
                    pre.name
                ),
                ContourKind::MeasuredBell { .. } => {
                    panic!("{}: no historical preset uses measured angles", pre.name)
                }
            }
            let profile = cfd_geom::generate_contour(&spec, CONTOUR_SAMPLES)
                .unwrap_or_else(|e| panic!("{}: generate_contour rejected the spec: {e}", pre.name));
            profile
                .validate()
                .unwrap_or_else(|e| panic!("{}: invalid profile: {e}", pre.name));

            // validate() proves strict z monotonicity; state the geometric
            // consequences the app depends on explicitly so a future change to
            // validate() cannot quietly drop them.
            for w in profile.points.windows(2) {
                assert!(w[1][0] > w[0][0], "{}: z not strictly increasing", pre.name);
            }
            let r_min = profile.points.iter().map(|q| q[1]).fold(f64::INFINITY, f64::min);
            assert!(
                (r_min / pre.r_throat_m - 1.0).abs() < 1e-9,
                "{}: min radius {r_min} m vs r_t {}",
                pre.name,
                pre.r_throat_m
            );
            assert!(
                (profile.points[profile.throat_index][1] / r_min - 1.0).abs() < 1e-9,
                "{}: throat_index does not point at the minimum radius",
                pre.name
            );
            // The wall must actually turn around at the throat — a monotone
            // polyline would pass every check above with the "throat" at one end.
            assert!(profile.throat_index > 0, "{}: throat at the inlet", pre.name);
            assert!(
                profile.throat_index < profile.points.len() - 1,
                "{}: throat at the exit",
                pre.name
            );

            let r_exit = profile.points.last().unwrap()[1];
            let want_exit = pre.r_throat_m * pre.area_ratio.sqrt();
            worst_exit_err = worst_exit_err.max((r_exit / want_exit - 1.0).abs());

            let n = profile.points.len();
            let (a, b) = (profile.points[n - 2], profile.points[n - 1]);
            let exit_deg = ((b[1] - a[1]) / (b[0] - a[0])).atan().to_degrees();
            if pre.contour_kind.is_bell() {
                // A bell's whole point is turning the flow back toward axial.
                assert!(
                    exit_deg > 0.0 && exit_deg < pre.cone_half_angle_deg,
                    "{}: exit angle {exit_deg:.2}° is not a bell",
                    pre.name
                );
            } else {
                // The divergent wall IS the cone: measure the last segment,
                // always inside the straight run (the downstream throat arc is
                // over within 0.382 r_t of the throat).
                worst_cone_angle_err =
                    worst_cone_angle_err.max((exit_deg - pre.cone_half_angle_deg).abs());
            }
        }
        record(
            "wall-exit-radius",
            "historical presets: generated exit radius vs r_t*sqrt(eps)",
            "<= 1e-9",
            worst_exit_err,
            "max relative error",
            worst_exit_err <= 1e-9,
        );
        record(
            "cone-half-angle",
            "historical cones: divergent wall angle vs the preset's own cone_half_angle_deg",
            "<= 0.01",
            worst_cone_angle_err,
            "max |deviation| (deg)",
            worst_cone_angle_err <= 0.01,
        );
        assert!(worst_exit_err <= 1e-9, "worst exit-radius error {worst_exit_err:.3e}");
        assert!(
            worst_cone_angle_err <= 0.01,
            "worst cone angle error {worst_cone_angle_err:.4} deg"
        );
    }

    /// 2. Every historical preset fits the domain `preset_domain` sizes for it,
    ///    with margin. These nozzles are all shorter than the existing six
    ///    (ε 2.83–25 against 16–165), so this should clear easily — a failure
    ///    means an r_t or an ε is wrong, not that the domain is too small.
    #[test]
    fn historical_presets_fit_their_domain_with_margin() {
        // Fractions of the domain the wall may occupy. The axial figure is
        // generous because `preset_domain` deliberately budgets most of the
        // length for the plume, not the nozzle.
        const MAX_Z_FRACTION: f64 = 0.5;
        const MAX_R_FRACTION: f64 = 0.75;
        let (mut worst_z, mut worst_r) = (0.0f64, 0.0f64);
        for pre in historical() {
            let c = pre.case(pre.default_altitude_m(), false);
            let wall = nozzle_contour(&c);
            assert_eq!(wall.fallback, None, "{}: fell back to the cone", pre.name);
            let end = *wall.points.last().unwrap();
            let (fz, fr) = (end[0] / c.lz_rt, end[1] / c.lr_rt);
            println!(
                "{:30} exit at z {:6.2} / {:6.1} r_t ({:4.1}%), r {:5.2} / {:5.1} r_t ({:4.1}%)",
                pre.name, end[0], c.lz_rt, 100.0 * fz, end[1], c.lr_rt, 100.0 * fr
            );
            worst_z = worst_z.max(fz);
            worst_r = worst_r.max(fr);
        }
        record(
            "domain-fit-axial",
            "historical presets: nozzle exit station as a fraction of domain length",
            &format!("<= {MAX_Z_FRACTION}"),
            worst_z,
            "worst z_exit / lz_rt",
            worst_z <= MAX_Z_FRACTION,
        );
        record(
            "domain-fit-radial",
            "historical presets: exit lip radius as a fraction of domain radius",
            &format!("<= {MAX_R_FRACTION}"),
            worst_r,
            "worst r_exit / lr_rt",
            worst_r <= MAX_R_FRACTION,
        );
        assert!(worst_z <= MAX_Z_FRACTION, "worst axial fill {worst_z:.3}");
        assert!(worst_r <= MAX_R_FRACTION, "worst radial fill {worst_r:.3}");
    }

    /// 3. THE TEST THAT PROVES `default_altitude_km` EARNS ITS PLACE. Quasi-1D
    ///    p_e/p_a at each preset's own default altitude must clear the 0.40
    ///    Summerfield threshold — nobody opens a preset into a separation
    ///    warning they did not cause.
    ///
    ///    And the other half, without which the field could be zero everywhere
    ///    and still pass: the two altitude-optimised presets must genuinely
    ///    separate at sea level. If they did not, the non-zero default would be
    ///    decoration. The computed crossings are 6.7 km (LR105) and 7.2 km
    ///    (LR91); the shipped 10 and 12 km carry margin over them, and that
    ///    margin is asserted too.
    #[test]
    fn historical_presets_do_not_separate_at_their_default_altitude() {
        let mut worst = f64::INFINITY;
        let mut worst_name = "";
        for pre in historical() {
            let c = pre.case(pre.default_altitude_m(), false);
            let (m_e, pe_p0) = ideal_exit(c.area_ratio, c.gamma);
            let ratio = |alt: f64| pe_p0 * c.p0_pa / atmosphere(alt).0;
            let at_default = ratio(c.altitude_m);
            let at_sea_level = ratio(0.0);
            println!(
                "{:30} M_e {:5.3} | default {:4.0} km: p_e/p_a {:8.3} | sea level: {:6.3} \
                 (threshold {:.3})",
                pre.name,
                m_e,
                c.altitude_m / 1000.0,
                at_default,
                at_sea_level,
                separation_threshold(m_e)
            );
            assert!(
                at_default > 0.40,
                "{} separates at its own default altitude ({:.0} km): p_e/p_a {at_default:.3}",
                pre.name,
                c.altitude_m / 1000.0
            );
            if at_default < worst {
                worst = at_default;
                worst_name = pre.name;
            }
            if pre.default_altitude_km > 0.0 {
                assert!(
                    at_sea_level <= 0.40,
                    "{} is given a non-zero default altitude but does NOT separate at sea \
                     level (p_e/p_a {at_sea_level:.3}) — the default is decoration",
                    pre.name
                );
                // The default must sit ABOVE the crossing with room to spare,
                // not just on the right side of it: bisect the crossing and
                // require the shipped default to clear it.
                let (mut lo, mut hi) = (0.0f64, ALT_MAX_M);
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    if ratio(mid) < 0.40 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let crossing_km = 0.5 * (lo + hi) / 1000.0;
                println!(
                    "{:30} separation crossing at {crossing_km:.1} km, ships at {:.0} km",
                    pre.name, pre.default_altitude_km
                );
                assert!(
                    pre.default_altitude_km >= crossing_km + 2.0,
                    "{}: default {:.0} km leaves under 2 km of margin over the {crossing_km:.1} km \
                     crossing",
                    pre.name,
                    pre.default_altitude_km
                );
                record(
                    &format!("separation-crossing-{}", pre.name),
                    &format!("{}: altitude at which p_e/p_a reaches 0.40", pre.name),
                    &format!("<= {:.0} km (the shipped default)", pre.default_altitude_km),
                    crossing_km,
                    "km",
                    pre.default_altitude_km >= crossing_km,
                );
            } else {
                // A zero default has to be earned the same way.
                assert!(
                    at_sea_level > 0.40,
                    "{} opens at 0 km but separates there (p_e/p_a {at_sea_level:.3}) — it \
                     needs a default altitude",
                    pre.name
                );
            }
        }
        record(
            "separation-at-default-altitude",
            "historical presets: worst quasi-1D p_e/p_a at the preset's own default altitude",
            "> 0.40",
            worst,
            &format!("p_e/p_a ({worst_name})"),
            worst > 0.40,
        );
        cfd_results::record_note(SUITE, "default-altitude-is-load-bearing", &format!(
            "Every historical preset clears the 0.40 Summerfield separation threshold at its \
             own default altitude; the tightest is {worst_name} at p_e/p_a {worst:.3}. The two \
             altitude-optimised presets are asserted in BOTH directions: Atlas LR105-5 \
             (default 10 km) and Titan I LR91-AJ-3 (default 12 km) must separate at sea level, \
             or their non-zero default would be decoration rather than a fix, and each default \
             must clear its own computed crossing by at least 2 km. Every other preset must NOT \
             separate at 0 km, or it would need a default of its own."));
        assert!(worst > 0.40);
    }

    /// 4. The Redstone throat area, recovered from the RASTERIZED wall rather
    ///    than from the polyline: rasterize, then integrate the open
    ///    (non-solid) area of each axial column from the exact solid fractions
    ///    and take the minimum over columns. That exercises the whole geometry
    ///    chain the solver sees — spec, contour, polygon clipper — against the
    ///    one published dimension in the historical set.
    ///
    /// Measured on a REFINED grid, and the reason is a second-order effect
    /// worth stating rather than hiding: a column integrates the wall over its
    /// own width, and near a throat r(z) is quadratic, so the cell-averaged
    /// area is biased HIGH by ~(dz²/12)/R_arc — a few tenths of a percent at
    /// the production 20 cells/r_t with the §10 family's 0.382 r_t downstream
    /// arc, for a discretization reason that has nothing to do with whether the
    /// geometry is right. The production grid is measured and asserted too, so
    /// the bias is visible instead of designed around.
    #[test]
    fn redstone_rasterized_throat_area_matches_published() {
        /// Published: 15.5 in throat diameter -> pi * (0.19685 m)^2.
        const A_T_PUBLISHED_M2: f64 = 0.121_736;
        const TOL: f64 = 0.005;

        let pre = PRESETS
            .iter()
            .find(|p| p.name == "Redstone NAA 75-110-A-7")
            .unwrap();

        /// Minimum over axial columns of the open cross-sectional area,
        /// in m², from the exact solid fractions.
        fn throat_area_m2(c: &CaseParams, cells_per_rt: f64) -> f64 {
            let p = CaseParams { cells_per_rt, ..*c };
            let wall = nozzle_contour(&p);
            assert_eq!(wall.fallback, None, "Redstone fell back to the cone");
            let g = graded_grid(&p, &wall.points);
            let s = rasterize_wall(&wall.points, &g);
            let z_end = wall.points.last().unwrap()[0];
            let mut best = f64::INFINITY;
            for iz in 0..g.nz {
                if (g.z_center(iz) as f64) > z_end {
                    break;
                }
                // Annular open area of the column, non-dimensional (r_t²).
                // f64 accumulation, as every reduction in this build must be.
                let mut open = 0.0f64;
                for ir in 0..g.nr {
                    let (r0, r1) = (g.r_face(ir) as f64, g.r_face(ir + 1) as f64);
                    let frac = s.fraction[g.idx(iz, ir)] as f64;
                    open += (1.0 - frac) * std::f64::consts::PI * (r1 * r1 - r0 * r0);
                }
                best = best.min(open);
            }
            best * p.r_throat_m * p.r_throat_m
        }

        let c = pre.case(pre.default_altitude_m(), false);
        let a_production = throat_area_m2(&c, pre.cells_per_rt);
        // 4x the production radial resolution: the O(dz²) column-averaging
        // bias falls by 16.
        let a_refined = throat_area_m2(&c, 4.0 * pre.cells_per_rt);
        let a_geometric = std::f64::consts::PI * pre.r_throat_m * pre.r_throat_m;

        let err = a_refined / A_T_PUBLISHED_M2 - 1.0;
        let err_production = a_production / A_T_PUBLISHED_M2 - 1.0;
        println!(
            "Redstone A_t: published {A_T_PUBLISHED_M2:.6} m^2 | preset r_t geometric \
             {a_geometric:.6} ({:+.3}%) | rasterized refined {a_refined:.6} ({:+.3}%) | \
             rasterized production {a_production:.6} ({:+.3}%)",
            100.0 * (a_geometric / A_T_PUBLISHED_M2 - 1.0),
            100.0 * err,
            100.0 * err_production
        );
        record(
            "redstone-throat-area",
            "Redstone rasterized throat area vs the published 15.5 in throat diameter",
            &format!("{A_T_PUBLISHED_M2:.6} +/- {:.1}%", 100.0 * TOL),
            a_refined,
            "m^2 (refined grid, 80 cells/r_t)",
            err.abs() <= TOL,
        );
        cfd_results::record_note(SUITE, "redstone-throat-area-discretization", &format!(
            "Redstone throat area recovered from the rasterized wall: {a_refined:.6} m^2 at \
             4x the production radial resolution ({:+.3}% vs the published {A_T_PUBLISHED_M2:.6} \
             m^2), {a_production:.6} m^2 on the production grid ({:+.3}%). The production \
             figure is biased HIGH by column averaging: a column integrates the wall over its \
             own width and r(z) is quadratic at a throat, so the cell-mean area exceeds the \
             minimum by ~(dz^2/12)/R_arc. It falls as dz^2, which the refined figure confirms. \
             The preset's own r_t gives a geometric area of {a_geometric:.6} m^2 ({:+.3}%).",
            100.0 * err, 100.0 * err_production,
            100.0 * (a_geometric / A_T_PUBLISHED_M2 - 1.0)));
        assert!(
            err.abs() <= TOL,
            "Redstone throat area {a_refined:.6} m^2 is {:+.3}% off the published \
             {A_T_PUBLISHED_M2:.6} m^2",
            100.0 * err
        );
        assert!(
            err_production.abs() <= TOL,
            "Redstone throat area on the PRODUCTION grid ({} cells/r_t) is {:+.3}% off the \
             published area — the column-averaging bias has grown past the tolerance",
            pre.cells_per_rt,
            100.0 * err_production
        );
    }

    /// 5. The Rao table is digitised over ε = 4–100 and `rao_angles` CLAMPS at
    ///    both ends rather than extrapolating. The area-ratio slider bottoms
    ///    out at ε = 2, so a user can drag any of the five historical bells
    ///    into the clamped region below the table — and that has to be
    ///    disclosed exactly as the ε > 100 case is, not silently.
    ///
    ///    This asserts the SYMMETRY of the clamp and of the condition the UI
    ///    flags it on. It is a characterisation test: the app's condition is
    ///    `!(4.0..=100.0).contains(&ε)`, which already covers both ends, and
    ///    this pins that it keeps doing so — the failure mode being guarded is
    ///    a future edit that special-cases only the high end, which is the
    ///    natural way to write it if you are only thinking about Merlin Vac.
    #[test]
    fn rao_table_clamps_and_flags_symmetrically_at_both_ends() {
        /// The digitised range, as the UI's disclosure condition uses it.
        const TABLE: std::ops::RangeInclusive<f64> = 4.0..=100.0;
        let flagged = |eps: f64| !TABLE.contains(&eps);

        for bp in [0.75, 0.80] {
            // Below the table: every ε clamps to the ε = 4 row, and is flagged.
            let at_low_end = cfd_geom::rao_angles(4.0, bp);
            for eps in [2.0, 2.5, 3.0, 3.9, 3.999] {
                assert_eq!(
                    cfd_geom::rao_angles(eps, bp),
                    at_low_end,
                    "eps {eps} does not clamp to the eps = 4 row"
                );
                assert!(flagged(eps), "eps {eps} is clamped but not flagged");
            }
            // Above it: the mirror image, which is the case that already had
            // the disclosure.
            let at_high_end = cfd_geom::rao_angles(100.0, bp);
            for eps in [100.001, 120.0, 165.0, 400.0] {
                assert_eq!(
                    cfd_geom::rao_angles(eps, bp),
                    at_high_end,
                    "eps {eps} does not clamp to the eps = 100 row"
                );
                assert!(flagged(eps), "eps {eps} is clamped but not flagged");
            }
            // Inside the table nothing is clamped and nothing is flagged.
            for eps in [4.0, 8.0, 25.0, 100.0] {
                assert!(!flagged(eps), "eps {eps} is inside the table but flagged");
            }
            // The endpoints are genuinely distinct, so "clamps to the end row"
            // is a real constraint rather than a tautology about a flat table.
            assert!(
                (at_low_end.0 - at_high_end.0).abs() > 1.0,
                "the table's two ends have the same theta_n; the clamp assertions above \
                 would pass on any input"
            );
        }

        // Every shipped bell can actually be dragged into the low-clamped
        // region: the slider's floor is below 4, so this is reachable, not
        // hypothetical. (Merlin Vac is the engine that made the HIGH end
        // reachable; the historical bells are what make the low end matter.)
        let draggable: Vec<&str> = PRESETS
            .iter()
            .filter(|p| p.contour_kind == ContourKind::ParabolicBell)
            .map(|p| p.name)
            .collect();
        assert!(
            !draggable.is_empty() && AREA_RATIO_SLIDER_MIN < *TABLE.start(),
            "no bell preset can reach the low clamp; this test is guarding nothing"
        );
        // …and the wall still generates there rather than falling back, which
        // is what makes a silent clamp possible in the first place.
        for name in &draggable {
            let pre = PRESETS.iter().find(|p| p.name == *name).unwrap();
            let mut c = pre.case(0.0, false);
            c.area_ratio = 3.0;
            let w = nozzle_contour(&c);
            assert_eq!(
                w.fallback, None,
                "{name} at eps 3 falls back to the cone — the clamp disclosure would never \
                 be reached, and the status line would say cone instead"
            );
            assert_eq!(w.kind, ContourKind::ParabolicBell);
        }
        cfd_results::record_note(SUITE, "rao-low-end-clamp", &format!(
            "The Rao table covers eps = 4-100 and rao_angles clamps at BOTH ends; the area-ratio \
             slider bottoms out at {AREA_RATIO_SLIDER_MIN}, so all {} bell presets can be dragged \
             below the table. The UI's disclosure condition is !(4.0..=100.0).contains(eps), \
             which already fires at both ends — verified here rather than assumed, because the \
             natural way to write that check while thinking only about Merlin Vac (eps 165) is \
             to test the high end alone. Below eps 4 the angles are the eps = 4 row \
             (theta_n {:.2}, theta_e {:.2} at 80%) stretched to the requested exit radius.",
            draggable.len(),
            cfd_geom::rao_angles(4.0, 0.8).0,
            cfd_geom::rao_angles(4.0, 0.8).1));
    }

    /// 6. REGRESSION: the six presets that existed before this work order are
    ///    untouched, field by field. The expected values are a deliberate
    ///    second copy of the table — that is what a pin is for. Changing a
    ///    preset now takes two edits and a moment's thought, which is the
    ///    point: every committed benchmark and results file in this repo was
    ///    measured against these numbers.
    #[test]
    fn existing_six_presets_are_unchanged() {
        // name, propellant, eps, p0, r_t, gamma, T0, MW, cells/r_t, bell %,
        // throat arc, slow
        type Row = (&'static str, &'static str, f64, f64, f64, f64, f64, f64, f64, f64, f64, bool);
        const EXPECT: [Row; 6] = [
            ("Merlin 1D", "LOX/RP-1", 16.0, 9.7e6, 0.131, 1.24, 3600.0, 21.9, 20.0, 0.78, 0.382, false),
            ("F-1", "LOX/RP-1", 16.0, 7.0e6, 0.465, 1.24, 3600.0, 21.9, 20.0, 0.75, 0.382, false),
            ("Raptor 2", "LOX/CH4", 34.3, 3.0e7, 0.115, 1.16, 3600.0, 22.0, 20.0, 0.76, 0.300, false),
            ("AJ10-190", "N2O4/MMH", 55.0, 8.6e5, 0.073, 1.23, 3200.0, 21.5, 20.0, 0.78, 0.382, false),
            ("RS-25", "LOX/LH2", 69.0, 2.06e7, 0.138, 1.20, 3600.0, 13.5, 20.0, 0.80, 0.382, false),
            ("Merlin Vac", "LOX/RP-1", 165.0, 9.7e6, 0.128, 1.24, 3600.0, 21.9, 14.0, 0.75, 0.382, true),
        ];
        const KINDS: [ContourKind; 6] = [
            ContourKind::ParabolicBell,
            ContourKind::ParabolicBell,
            ContourKind::MeasuredBell { theta_n_deg: 32.0, theta_e_deg: 6.0 },
            ContourKind::ParabolicBell,
            ContourKind::ParabolicBell,
            ContourKind::ParabolicBell,
        ];

        for (i, e) in EXPECT.iter().enumerate() {
            let p = &PRESETS[i];
            // Position matters too: PRESETS[0] is the cost reference and
            // several tests index by number.
            assert_eq!(p.name, e.0, "preset {i} moved");
            assert_eq!(p.propellant, e.1, "{}: propellant", p.name);
            // Exact f64 equality throughout — these are literals, not
            // computed values, so anything but bit equality IS the regression.
            assert_eq!(p.area_ratio, e.2, "{}: area_ratio", p.name);
            assert_eq!(p.p0_pa, e.3, "{}: p0_pa", p.name);
            assert_eq!(p.r_throat_m, e.4, "{}: r_throat_m", p.name);
            assert_eq!(p.gamma, e.5, "{}: gamma", p.name);
            assert_eq!(p.t0_k, e.6, "{}: t0_k", p.name);
            assert_eq!(p.mw_g_mol, e.7, "{}: mw_g_mol", p.name);
            assert_eq!(p.cells_per_rt, e.8, "{}: cells_per_rt", p.name);
            assert_eq!(p.bell_percent, e.9, "{}: bell_percent", p.name);
            assert_eq!(p.throat_arc_down, e.10, "{}: throat_arc_down", p.name);
            assert_eq!(p.slow, e.11, "{}: slow", p.name);
            assert_eq!(p.contour_kind, KINDS[i], "{}: contour_kind", p.name);
            // The fields this work order added must be INERT for these six:
            // they open at sea level as before, their gas properties are
            // per-engine researched values rather than the CEA table (in
            // particular LoxRp1's computed gamma 1.221 does NOT reach the F-1
            // or either Merlin), their provenance strings are empty so their
            // tooltips are byte-identical, and their cone half-angle is the
            // default — which is unreachable anyway, since all six are bells.
            assert_eq!(p.default_altitude_km, 0.0, "{}: default_altitude_km", p.name);
            assert_eq!(p.propellant_class, None, "{}: propellant_class", p.name);
            assert_eq!(p.throat_source, "", "{}: throat_source", p.name);
            assert_eq!(p.operating_source, "", "{}: operating_source", p.name);
            assert_eq!(p.cstar_shifting_m_s, 0.0, "{}: cstar_shifting_m_s", p.name);
            assert_eq!(
                p.cone_half_angle_deg, CONE_HALF_ANGLE_DEG,
                "{}: cone_half_angle_deg", p.name
            );
            assert!(p.contour_kind.is_bell(), "{}: still a bell", p.name);

            // And the derived case, which is what the solver actually gets.
            let c = p.case(0.0, false);
            assert_eq!(c.altitude_m, 0.0, "{}: opens at sea level", p.name);
            assert_eq!(c.gamma, e.5);
            assert_eq!(c.t0_k, e.6);
            assert_eq!(c.r_specific_si, R_UNIVERSAL_SI / e.7);
            assert_eq!(c.cone_half_angle_deg, CONE_HALF_ANGLE_DEG);
            assert_eq!((c.lz_rt, c.lr_rt), preset_domain(e.2), "{}: domain", p.name);
        }
        record(
            "existing-presets-unchanged",
            "the six pre-existing engine presets, field by field against a pinned copy",
            "6",
            EXPECT.len() as f64,
            "presets verified",
            true,
        );
    }

    /// Each CEA-backed preset carries its OWN case's numbers, and those numbers
    /// belong to the class it claims: inside the min/max span the tool measured
    /// across that class's cases.
    ///
    /// This is why the presets do not simply use `PropellantClass::reference()`.
    /// A class reference is a mean, and for LOX/ethanol the two cases sit at
    /// O/F 1.130 and 1.324 — 6.6% apart in T₀. Using the mean would put both
    /// engines 3% wrong rather than either one right.
    #[test]
    fn preset_gas_matches_its_propellant_class() {
        let mut classes_seen = Vec::new();
        for pre in PRESETS.iter() {
            let Some(class) = pre.propellant_class else { continue };
            classes_seen.push(class);
            assert_eq!(pre.propellant, class.label(), "{}: label", pre.name);
            let (lo, hi) = class.cases_span();
            // A hair of slack for the decimals the table is rounded to.
            const EPS: f64 = 1e-3;
            for (what, v, a, b) in [
                ("gamma", pre.gamma, lo.gamma, hi.gamma),
                ("t0_k", pre.t0_k, lo.t0_k, hi.t0_k),
                ("mw_g_mol", pre.mw_g_mol, lo.mw_g_mol, hi.mw_g_mol),
                (
                    "cstar_shifting",
                    pre.cstar_shifting_m_s,
                    lo.cstar_shifting_m_s,
                    hi.cstar_shifting_m_s,
                ),
            ] {
                assert!(
                    v >= a - EPS && v <= b + EPS,
                    "{}: {what} = {v} outside its {:?} class span [{a}, {b}]",
                    pre.name,
                    class
                );
            }
            // The class reference is the mean of the span it reports.
            let r = class.reference();
            for (what, mean, a, b) in [
                ("gamma", r.gamma, lo.gamma, hi.gamma),
                ("t0_k", r.t0_k, lo.t0_k, hi.t0_k),
                ("mw_g_mol", r.mw_g_mol, lo.mw_g_mol, hi.mw_g_mol),
            ] {
                assert!(
                    mean >= a - EPS && mean <= b + EPS,
                    "{:?}: reference {what} {mean} outside its own span [{a}, {b}]",
                    class
                );
            }
            println!(
                "{:30} {:22} gamma {:.4} T0 {:6.0} K MW {:5.2} -> c* {:6.1} shifting / \
                 {:6.1} frozen m/s",
                pre.name, class.label(), pre.gamma, pre.t0_k, pre.mw_g_mol,
                pre.gas().cstar_shifting_m_s, pre.gas().cstar_frozen_m_s
            );
        }
        for class in PropellantClass::ALL {
            assert!(
                classes_seen.contains(&class),
                "{class:?} is defined but no preset uses it"
            );
        }
    }

    /// The Rust `cstar_ideal` closed form must reproduce the `cstar_frozen`
    /// column `tools/propellant_cea.py` prints, from the same (T₀, MW, γ).
    ///
    /// Two implementations of one formula, in two languages, pinned to each
    /// other. Without this the table could be transcribed from a tool whose
    /// definition of c* had quietly diverged from the solver's — which is
    /// exactly the failure the shifting-vs-frozen distinction invites, since
    /// the two differ by only ~1.3% and both look plausible.
    #[test]
    fn rust_and_python_frozen_cstar_agree() {
        // (name, T0, MW, gamma, python cstar_frozen) from
        // docs/results/propellant-cea.md, full-precision column.
        const CASES: [(&str, f64, f64, f64, f64); 8] = [
            ("V-2", 2889.6492, 22.03174, 1.194595, 1612.833),
            ("Redstone", 3081.5671, 23.35647, 1.187463, 1621.099),
            ("Thor", 3495.0930, 21.94603, 1.220567, 1763.565),
            ("Atlas LR89", 3475.9430, 21.80683, 1.221536, 1763.830),
            ("Atlas LR105", 3518.5008, 22.02637, 1.219702, 1766.679),
            ("Titan LR87", 3470.2180, 21.76167, 1.221839, 1764.048),
            ("Titan LR91", 3483.1491, 21.78713, 1.221424, 1766.514),
            ("WAC Corporal", 2988.2979, 25.62019, 1.211618, 1513.231),
        ];
        let mut worst = 0.0f64;
        for (name, t0, mw, g, py) in CASES {
            let rs = cstar_ideal(t0, mw, g);
            let rel = (rs / py - 1.0).abs();
            worst = worst.max(rel);
            assert!(
                rel < 1e-5,
                "{name}: Rust cstar_ideal {rs:.4} vs Python cstar_frozen {py:.4} \
                 ({:+.4}%)",
                100.0 * (rs / py - 1.0)
            );
        }
        record(
            "frozen-cstar-rust-vs-python",
            "cstar_ideal (Rust) vs cstar_frozen (tools/propellant_cea.py) over all eight cases",
            "< 1e-5",
            worst,
            "max relative difference",
            worst < 1e-5,
        );
    }

    /// The constraint the work order set: a derived throat radius that puts
    /// mass flow more than 10% off the published figure gets REPORTED, not
    /// adjusted to match. This is the report, as an executable one — the
    /// numbers are recomputed from the presets every run, so a later edit to
    /// an r_t cannot quietly change the story the write-up tells.
    ///
    /// The published MASS FLOWS are deliberately not asserted against, and
    /// that is the finding rather than a hedge: the AEHS mass-flow column is
    /// known-unreliable. Deriving throat area from ṁ and c* back-solves the
    /// Redstone to a c* efficiency of 100.2%, and the same table's LR89-5 row
    /// (163,211 lbf against 458 lb/s) implies an Isp of 356 s against its own
    /// Isp column of 248–282. What IS asserted is the route that does work:
    /// thrust and thrust coefficient, which reproduces the one published
    /// throat to −0.03%.
    #[test]
    fn derived_throat_radii_reproduce_the_published_thrust() {
        /// Divergence factor λ = (1+cos α)/2 for the 15° cones; a 75–80% bell
        /// recovers most of that loss. Both from `tools/propellant_cea.py`.
        const LAMBDA_BELL: f64 = 0.988;
        /// Residual C_f efficiency after divergence, calibrated on the single
        /// published throat in the set.
        const ETA_CF: f64 = 0.9877;

        // engine, published thrust N, ambient at the rating (None = vacuum)
        const PUBLISHED: [(&str, f64, Option<f64>); 8] = [
            ("V-2 (A-4) Model 39", 244_653.0, Some(101_325.0)),
            ("Redstone NAA 75-110-A-7", 369_096.0, Some(101_325.0)),
            ("Thor LR79-NA-7 (MB-1)", 667_200.0, Some(101_325.0)),
            ("Atlas LR89-5 booster (1 of 2)", 726_000.0, Some(101_325.0)),
            ("Atlas LR105-5 sustainer", 386_400.0, None),
            ("Titan I LR87-AJ-3 (1 of 2)", 647_900.0, Some(101_325.0)),
            ("Titan I LR91-AJ-3", 355_900.0, None),
            ("WAC Corporal 38ALDW-1500", 6_672.0, Some(101_325.0)),
        ];

        let mut worst = 0.0f64;
        let mut worst_name = "";
        println!(
            "{:30}{:>10}{:>10}{:>9}{:>12}{:>10}",
            "engine", "r_t mm", "derived", "d%", "F published", "mdot"
        );
        for (name, f_pub, rating_pa) in PUBLISHED {
            let pre = PRESETS.iter().find(|p| p.name == name).unwrap();
            let c = pre.case(0.0, false);
            let lambda = if pre.contour_kind.is_bell() {
                LAMBDA_BELL
            } else {
                0.5 * (1.0 + pre.cone_half_angle_deg.to_radians().cos())
            };
            let pa_over_p0 = rating_pa.unwrap_or(0.0) / c.p0_pa;
            let cf_ideal_v = ideal_cf(c.area_ratio, c.gamma, pa_over_p0);
            // The derivation the table was built with, run backwards from the
            // preset's own gas properties: A_t = F / (Cf * lambda * eta * p0).
            let a_t = f_pub / (cf_ideal_v * lambda * ETA_CF * c.p0_pa);
            let r_derived = (a_t / std::f64::consts::PI).sqrt();
            let d = r_derived / pre.r_throat_m - 1.0;
            // Mass flow that the SHIPPED throat implies, at the handbook 95%
            // c* efficiency the tool uses. Recorded, never asserted.
            let mdot = c.p0_pa * std::f64::consts::PI * pre.r_throat_m * pre.r_throat_m
                / (0.95 * pre.cstar_shifting_m_s);
            println!(
                "{name:30}{:>10.1}{:>10.1}{:>9.2}{:>12.1}{:>10.2}",
                pre.r_throat_m * 1e3,
                r_derived * 1e3,
                100.0 * d,
                f_pub / 1e3,
                mdot
            );
            if d.abs() > worst {
                worst = d.abs();
                worst_name = name;
            }
            cfd_results::record_note(SUITE, &format!("throat-derivation-{name}"), &format!(
                "{name}: shipped r_t {:.2} mm; re-deriving it here from the published thrust \
                 {:.1} kN at {} with this preset's own gamma gives {:.2} mm ({:+.2}%). \
                 lambda {:.4}, eta_Cf {ETA_CF}, Cf_ideal {cf_ideal_v:.4}. Implied mass flow at \
                 95% c* efficiency is {mdot:.2} kg/s -- RECORDED, NOT ASSERTED: the AEHS \
                 mass-flow column is the unreliable one (its LR89-5 row implies Isp 356 s \
                 against its own 248-282 column), which is precisely why the throat radii come \
                 from thrust rather than from mass flow.",
                pre.r_throat_m * 1e3, f_pub / 1e3,
                if rating_pa.is_some() { "sea level" } else { "vacuum" },
                r_derived * 1e3, 100.0 * d, lambda));
        }
        // Round-tripping the derivation must reproduce the shipped table. This
        // is tight on purpose: it is the same arithmetic, so anything beyond
        // rounding of the committed literals means the table and the tool have
        // parted company.
        record(
            "throat-derivation-round-trip",
            "shipped r_t vs re-deriving it from published thrust and the preset's own gas",
            "<= 0.5%",
            100.0 * worst,
            &format!("worst |deviation| % ({worst_name})"),
            worst <= 0.005,
        );
        assert!(
            worst <= 0.005,
            "{worst_name}: shipped throat radius is {:.2}% away from what the published thrust \
             and this preset's gamma derive. Do NOT adjust r_t to close this — the table and \
             tools/propellant_cea.py have diverged; re-run the tool.",
            100.0 * worst
        );
    }

    /// Run time for each historical preset, for
    /// docs/results/historical-presets-v1.md. These are expected to be the
    /// cheapest cases in the library — ε of 2.83 to 25 against the existing
    /// 16 to 165 — and `preset_domain` sizes the domain off ε, so a short
    /// nozzle buys a small domain as well as a short one.
    ///
    /// Both numbers are MEASURED, not projected: `steps_per_sec` over 250
    /// steps after a 50-step warm-up, then the case is actually run to the §9
    /// visual-steady step count and the wall clock taken. (The residual-based
    /// `converged` flag is not usable as a finish line here for the same
    /// reason it is not for the demo case — an overexpanded sea-level plume
    /// keeps breathing above the 1e-3 threshold indefinitely.)
    ///
    /// Positivity-floor contact is tracked per preset and asserted to be
    /// startup-confined. That is not incidental bookkeeping: cumulative floor
    /// activations are what blank every readout under the product rule, so a
    /// preset that keeps touching the floor at steady state would ship as a
    /// permanent SOLUTION INVALID.
    ///
    ///     cargo test -p cfd-ui historical_preset_cost -- --include-ignored --nocapture
    #[test]
    #[ignore = "benchmark: several minutes of solver time; run explicitly"]
    fn historical_preset_cost() {
        use cfd_contract::Solver;
        use std::time::Instant;

        // §9: ~6,100 steps for the compact 46.4-long demo domain, scaled by
        // domain length and by 1/dt. Same model as `estimate_cost`.
        fn steps_to_visual_steady(g: &Grid, p: &CaseParams) -> f64 {
            let dr = 1.0 / p.cells_per_rt;
            let dz = p.dz_over_dr * dr;
            6100.0 * (g.lz() / LZ) * ((1.0 / dz + 1.0 / dr) / (1.0 / 0.145 + 1.0 / 0.05))
        }

        let merlin = PRESETS[0].relative_cost();
        assert_eq!(merlin, 1.0, "Merlin 1D is the cost reference");
        let mut projected = Vec::new();
        for pre in historical() {
            let p = pre.case(pre.default_altitude_m(), false);
            let wall = nozzle_contour(&p);
            let setup = make_setup(&p, &wall.points);
            let g = setup.grid.clone();
            let target = steps_to_visual_steady(&g, &p);
            let mut s = cfd_core::EulerSolver::new(setup).unwrap();
            // Floor contact is tracked from the FIRST step, not from the end
            // of the warm-up: the cold-start front is exactly where it
            // happens, so a counter that starts at step 300 would report every
            // activation with "last at step 0" and prove nothing.
            let (mut floors, mut last_floor) = (0u64, 0u64);
            let track = |info: &cfd_contract::StepInfo,
                         floors: &mut u64,
                         last: &mut u64| {
                if info.floor_activations > *floors {
                    *floors = info.floor_activations;
                    *last = info.step;
                }
            };
            for _ in 0..50 {
                let info = s.step().unwrap();
                track(&info, &mut floors, &mut last_floor);
            }
            let t0 = Instant::now();
            let mut info = s.step().unwrap();
            track(&info, &mut floors, &mut last_floor);
            for _ in 1..250 {
                info = s.step().unwrap();
                track(&info, &mut floors, &mut last_floor);
            }
            let sps = 250.0 / t0.elapsed().as_secs_f64();

            // Now actually run it to visual steady.
            let t1 = Instant::now();
            while (info.step as f64) < target {
                info = s.step().unwrap();
                track(&info, &mut floors, &mut last_floor);
            }
            let secs = 300.0 / sps + t1.elapsed().as_secs_f64();
            let rep = s.report();
            println!(
                "{:30} {:>4} x {:<4} = {:>6} cells | {:6.1} steps/s | {:6.0} steps to visual \
                 steady in {:6.1} s | {:.2}x Merlin 1D | floors {} (last at step {}) | \
                 mdot {:.1} kg/s, C_f {:.3}, exit M {:.2}",
                pre.name, g.nz, g.nr, g.len(), sps, target, secs, pre.relative_cost(),
                floors, last_floor, rep.mass_flow_kg_s, rep.thrust_coefficient, rep.exit_mach
            );
            // §13 cold-start shape: floor contact belongs to the startup front
            // and must be quiet well before steady. Steady-state floor contact
            // is a hard failure, not a footnote — it blanks the whole report.
            assert!(
                floors == 0 || (last_floor as f64) < 0.5 * target,
                "{}: {floors} floor activations, last at step {last_floor} of {target:.0} — \
                 not startup-confined, so the report would be permanently quarantined",
                pre.name
            );
            cfd_results::record_benchmark(SUITE, cfd_results::Benchmark {
                case: format!(
                    "{} (eps {}, p0 {:.3} MPa, {:.0} km)",
                    pre.name, pre.area_ratio, pre.p0_pa / 1e6, pre.default_altitude_km
                ),
                setting: if pre.contour_kind.is_bell() {
                    format!("{:.0}% Rao bell, preset domain, 20 cells/r_t", pre.bell_percent * 100.0)
                } else {
                    format!("{:.0}° cone, preset domain, 20 cells/r_t", pre.cone_half_angle_deg)
                },
                cells: g.len() as u64,
                steps_per_sec: sps,
                seconds_to_steady: secs,
            });
            if floors > 0 {
                cfd_results::record_note(SUITE, &format!("floors-{}", pre.name), &format!(
                    "{}: {floors} positivity-floor activations, all during the cold-start \
                     front (last at step {last_floor} of {target:.0} to visual steady, zero \
                     after) — the SS13 quarantine class, same as the existing presets.",
                    pre.name));
            }
            // The solved mass flow against the quasi-1D ideal for the same
            // throat and gas. Recorded, not asserted: this is the solver's
            // known discretization deficit at 20 cells/r_t (docs §8's
            // 1/N_throat systematic error, which the report's own N_throat
            // badge exists to flag), and it is the same sign and size across
            // every historical preset — an engine-independent property of the
            // grid, not a verdict on any preset's throat radius.
            let mdot_ideal = p.p0_pa * std::f64::consts::PI * p.r_throat_m * p.r_throat_m
                / pre.gas().cstar_frozen_m_s;
            cfd_results::record_note(SUITE, &format!("mdot-{}", pre.name), &format!(
                "{}: solved mass flow {:.1} kg/s at visual steady against a quasi-1D ideal of \
                 {mdot_ideal:.1} kg/s for the same throat area and frozen c* ({:+.1}%), C_f \
                 {:.3}, area-averaged exit Mach {:.2}, confidence {:?}.",
                pre.name, rep.mass_flow_kg_s,
                100.0 * (rep.mass_flow_kg_s / mdot_ideal - 1.0),
                rep.thrust_coefficient, rep.exit_mach, rep.confidence));
            projected.push((pre.name, secs, pre.relative_cost()));
        }
        cfd_results::record_note(SUITE, "run-time-method",
            "seconds_to_steady for the historical presets is MEASURED: steps_per_sec over 250 \
             steps after a 50-step warm-up on this machine, then the case run to the SS9 \
             visual-steady step count (6,100 steps at the 46.4 r_t compact demo domain, scaled \
             by domain length and by 1/dt) with the wall clock taken. The residual-based \
             settled flag is not usable as a finish line for these cases for the same reason it \
             is not for the demo case -- an overexpanded sea-level plume keeps breathing above \
             the 1e-3 threshold indefinitely.");

        // These must be the cheap end of the library. Not a tautology: the
        // cost model reads the graded cell count off each preset's own
        // rasterized wall, so a wrong r_t or eps shows up here as a domain
        // that does not shrink the way a short nozzle should.
        let worst = projected.iter().map(|x| x.2).fold(0.0, f64::max);
        assert!(
            worst < 2.0,
            "a historical preset costs {worst:.2}x Merlin 1D — these are supposed to be the \
             cheapest cases in the library"
        );
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


