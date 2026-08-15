//! The demo case and everything the UI derives from it: the US standard
//! atmosphere behind the altitude slider, the parametric conical contour, a
//! stub rasterizer (exact-fraction rasterization is session C's; this band
//! rasterizer keeps the app alive until `cfd_geom::rasterize` lands), and the
//! quasi-1D helpers the honesty surface needs (separation threshold, ideal C_f).
//!
//! All wall geometry here is non-dimensional (units of throat radius); SI
//! appears only in `CaseParams` and the atmosphere.

use std::sync::Arc;

use cfd_contract::{Ambient, Chamber, GasModel, Grid, Numerics, RefScales, SolidField, SolveSetup};

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaseParams {
    pub p0_pa: f64,
    pub t0_k: f64,
    pub gamma: f64,
    pub r_specific_si: f64,
    pub r_throat_m: f64,
    pub area_ratio: f64,
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
        slow: false,
        note: "",
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
        slow: true,
        note: "The costliest preset by far even at the reduced 14 cells per \
               throat radius (the mass-flow badge goes amber for that reason) \
               — see the run-time multiple above.",
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
            let est = estimate_cost(&c, &conical_contour(c.area_ratio), None);
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

/// 15° conical nozzle contour per docs/physics-reference.md §10, as a sparse
/// control polyline (z, r) in r_t units: chamber wall at r = 2 (contraction
/// ratio 4), 30° converging cone, 1.5 r_t upstream arc, 0.382 r_t downstream
/// arc, straight cone to r_e = sqrt(area_ratio). Sparse on purpose — these
/// points double as the editor's control points.
pub fn conical_contour(area_ratio: f64) -> Vec<[f64; 2]> {
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
        assert!((costs[0] - 1.0).abs() < 1e-12);
        assert!((costs[1] - 1.0).abs() < 1e-12, "F-1 shares Merlin's domain");
        assert!(costs[2] > 1.5 && costs[3] > costs[2] && costs[4] > costs[3],
                "costs not increasing with area ratio: {costs:?}");
        assert!(costs[5] > costs[4] && costs[5] > 5.0, "Merlin Vac not costliest: {costs:?}");
        for p in PRESETS.iter() {
            let c = p.case(0.0, false);
            let pts = conical_contour(c.area_ratio);
            let end = pts.last().unwrap();
            assert!(end[0] < c.lz_rt, "{}: nozzle longer than domain", p.name);
            assert!(
                end[1] + 0.4 < c.lr_rt, // bell exit clears the domain top with margin
                "{}: bell exit outside domain",
                p.name
            );
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

    #[test]
    fn rasterized_throat_matches_the_contour() {
        let g = base_grid(&CaseParams::default());
        let pts = conical_contour(8.0);
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
            "throat {r_open_min}"
        );
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
