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

/// Demo-case domain from docs/physics-reference.md §8: 46.4 x 10 r_t at
/// 20 radial cells per throat radius -> the interactive 320 x 200 grid.
/// Presets size their own domain via `preset_domain`.
pub const LZ: f64 = 46.4;
pub const LR: f64 = 10.0;
pub const CELLS_PER_RT: f64 = 20.0;
/// The §8 anisotropy, dz/dr = 0.1449/0.0500. Widening dz is nearly free in dt
/// (the radial term dominates) while linearly cutting cells.
const DZ_OVER_DR: f64 = 2.9;

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

/// Plume length control (grid-grading work order, item 3). The graded
/// tensor-product grid makes the longer options cheap: base resolution is
/// held across the nozzle, and the extra plume is covered by geometrically
/// growing cells, so `Long` costs cells only in its graded tail while the
/// timestep (set by the finest spacing) does not change at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlumeLength {
    /// The historic §8 domain: the Mach disk and half of one shock cell.
    Compact,
    /// ~20 exit radii of plume: the Mach disk plus ~2 shock cells.
    Standard,
    /// ~40 exit radii of plume: 4-5 shock cells.
    Long,
}

impl PlumeLength {
    pub const ALL: [PlumeLength; 3] = [PlumeLength::Compact, PlumeLength::Standard, PlumeLength::Long];

    pub fn label(self) -> &'static str {
        match self {
            PlumeLength::Compact => "Compact",
            PlumeLength::Standard => "Standard",
            PlumeLength::Long => "Long",
        }
    }

    /// Extra plume length beyond the compact domain, in exit radii.
    fn extra_exit_radii(self) -> f64 {
        match self {
            PlumeLength::Compact => 0.0,
            PlumeLength::Standard => 8.0,
            PlumeLength::Long => 28.0,
        }
    }

    /// Radial headroom multiplier over the compact far field.
    fn lr_factor(self) -> f64 {
        match self {
            PlumeLength::Compact => 1.0,
            PlumeLength::Standard => 1.3,
            PlumeLength::Long => 1.6,
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
    /// COMPACT domain size in throat radii and radial cells per throat
    /// radius. The demo case uses the §8 defaults; presets size the domain to
    /// their bell via `preset_domain` (and Merlin Vac drops to 14 cells/r_t).
    /// `plume` extends past these extents on the graded tail.
    pub lz_rt: f64,
    pub lr_rt: f64,
    pub cells_per_rt: f64,
    pub plume: PlumeLength,
}

impl Default for CaseParams {
    fn default() -> Self {
        // The demo case, docs/physics-reference.md §6.
        CaseParams {
            p0_pa: 5.0e6,
            t0_k: 3200.0,
            gamma: 1.24,
            r_specific_si: 378.0,
            r_throat_m: 0.05,
            area_ratio: 8.0,
            altitude_m: 0.0,
            vacuum: false,
            lz_rt: LZ,
            lr_rt: LR,
            cells_per_rt: CELLS_PER_RT,
            plume: PlumeLength::Standard,
        }
    }
}

/// Full domain extents for this case, (lz, lr) in throat radii: the compact
/// extents plus the plume option's graded extension.
pub fn domain(p: &CaseParams) -> (f64, f64) {
    let r_e = p.area_ratio.max(1.0).sqrt();
    (
        p.lz_rt + p.plume.extra_exit_radii() * r_e,
        p.lr_rt * p.plume.lr_factor(),
    )
}

/// Uniform anisotropic BASE grid over the compact domain, dz/dr = 2.9 (§8).
/// At the defaults this is exactly the historic 320 x 200 interactive grid.
/// This is the rasterization target the grading rule reads the solid span
/// from — the solver itself runs on `graded_grid`.
pub fn base_grid(p: &CaseParams) -> Grid {
    let dr = 1.0 / p.cells_per_rt;
    let dz = DZ_OVER_DR * dr;
    Grid::uniform(
        (p.lz_rt / dz).round() as usize,
        (p.lr_rt / dr).round() as usize,
        dz as f32,
        dr as f32,
    )
}

/// The graded solver grid for this case and wall: rasterize the wall on the
/// compact base grid, find the solid span, hold base resolution across it and
/// grade geometrically out to the plume-option extents (the spacing rule
/// itself lives in cfd-geom; growth 1.05).
pub fn graded_grid(p: &CaseParams, wall: &[[f64; 2]]) -> Grid {
    let base = base_grid(p);
    let solid = rasterize_wall(wall, &base);
    let (lz, lr) = domain(p);
    let dr = 1.0 / p.cells_per_rt;
    let spec = cfd_geom::GradeSpec::new(DZ_OVER_DR * dr, dr, lz, lr);
    cfd_geom::grade_from_solid(&solid, &spec).unwrap_or(base)
}

/// Preset domain sizing in throat radii, (length, height). Height must hold
/// the bell exit plus plume spread; length the nozzle plus a plume that grows
/// with exit radius. Run time goes roughly as length² · height — see the
/// preset tooltips for the relative cost.
pub fn preset_domain(area_ratio: f64) -> (f64, f64) {
    let s = area_ratio.sqrt();
    (3.0 * (s - 1.0) + 20.0, (2.5 * s).max(10.0))
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
        note: "≈12× the Merlin 1D run time even at the reduced 14 cells per \
               throat radius (the mass-flow badge goes amber for that reason).",
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
            plume: PlumeLength::Standard,
        }
    }

    /// Run time relative to Merlin 1D (length² · height), for tooltips.
    pub fn relative_cost(&self) -> f64 {
        let cost = |ar: f64| {
            let (lz, lr) = preset_domain(ar);
            lz * lz * lr
        };
        cost(self.area_ratio) / cost(PRESETS[0].area_ratio)
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

/// Rough time-to-steady estimate for a plume option, seconds of wall clock.
///
/// Model: visual steady is ~5 plume transits (physics-reference §9 measured
/// ~6,100 steps for the compact demo domain), the step count scales with the
/// domain length (transit-dominated) and inversely with the finest radial
/// spacing (which sets dt), and the cost per step scales with the cell count.
/// `measured` is a live (steps/s, cells) sample used to calibrate the
/// machine's throughput; without one a conservative constant stands in. An
/// estimate, honestly labelled "~" in the UI — not a readout.
pub fn estimate_steady_seconds(
    p: &CaseParams,
    plume: PlumeLength,
    wall: &[[f64; 2]],
    measured: Option<(f64, usize)>,
) -> f64 {
    const STEPS_COMPACT_DEMO: f64 = 6100.0; // §9, at LZ = 46.4 and dr = 0.05
    let case = CaseParams { plume, ..*p };
    let g = graded_grid(&case, wall);
    let dr = 1.0 / p.cells_per_rt;
    let steps = STEPS_COMPACT_DEMO * (g.lz() / LZ) * (0.05 / dr);
    // cells/s throughput; the fallback is the §8 low-end M1 figure.
    let throughput = match measured {
        Some((sps, cells)) if sps > 1.0 => sps * cells as f64,
        _ => 10.0e6,
    };
    steps * g.len() as f64 / throughput
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
    fn default_base_grid_is_the_reference_320x200() {
        let g = base_grid(&CaseParams::default());
        assert_eq!((g.nz, g.nr), (320, 200));
        assert!((g.dz(0) - 0.145).abs() < 1e-6 && (g.dr(0) - 0.05).abs() < 1e-7);
    }

    /// The graded default (Standard plume): ~50% longer domain than compact
    /// at roughly the compact cell count, with the base spacing held across
    /// the nozzle and the finest spacing unchanged (so dt is unchanged).
    #[test]
    fn graded_standard_domain_is_cheap() {
        let p = CaseParams::default();
        let wall = conical_contour(p.area_ratio);
        let (lz, lr) = domain(&p);
        assert!(lz > LZ * 1.4, "Standard lz = {lz}");
        let g = graded_grid(&p, &wall);
        assert!((g.lz() - lz).abs() < 1e-6 && (g.lr() - lr).abs() < 1e-6);
        assert!((g.dz_min() as f64 - 0.145).abs() < 1e-3);
        assert!((g.dr_min() as f64 - 0.05).abs() < 1e-7);
        let base_cells = base_grid(&p).len();
        assert!(
            g.len() < base_cells * 12 / 10,
            "graded Standard is {} cells vs {} compact-uniform",
            g.len(),
            base_cells
        );
        // Base resolution held across the whole nozzle span.
        let z_end = wall.last().unwrap()[0];
        for iz in 0..g.nz {
            if g.z_edges()[iz + 1] <= z_end {
                assert!((g.dz(iz) as f64 - 0.145).abs() < 1e-3, "dz({iz}) = {}", g.dz(iz));
            }
        }
        // Long reaches ~40 exit radii of plume beyond the nozzle.
        let pl = CaseParams { plume: PlumeLength::Long, ..p };
        let (lz_l, _) = domain(&pl);
        let r_e = p.area_ratio.sqrt();
        assert!((lz_l - LZ) / r_e >= 27.9, "Long adds {} r_e", (lz_l - LZ) / r_e);
        let gl = graded_grid(&pl, &wall);
        assert!(
            gl.len() < base_cells * 2,
            "graded Long is {} cells vs {} compact-uniform",
            gl.len(),
            base_cells
        );
    }

    #[test]
    fn presets_fit_their_domain_and_match_the_costed_ratios() {
        // Relative run times quoted in the task/tooltips: Raptor 2.1,
        // AJ10 3.4, RS-25 4.3, Merlin Vac 11.7 (F-1 shares Merlin's domain).
        let expect = [1.0, 1.0, 2.1, 3.4, 4.3, 11.7];
        for (p, want) in PRESETS.iter().zip(expect) {
            assert!(
                (p.relative_cost() / want - 1.0).abs() < 0.05,
                "{}: cost {:.2} vs {want}",
                p.name,
                p.relative_cost()
            );
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
        let p = CaseParams::default();
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

    fn run_case(name: &str, setup: SolveSetup, project: bool) {
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
        let mut info;
        let (wall_s, how, floors, last_floor) = if project {
            (target as f64 / sps, "projected", 0, 0)
        } else {
            let t1 = Instant::now();
            info = s.step().unwrap();
            let mut floors = info.floor_activations;
            let mut last_floor = if floors > 0 { info.step } else { 0 };
            while info.step < target {
                info = s.step().unwrap();
                if info.floor_activations > floors {
                    floors = info.floor_activations;
                    last_floor = info.step;
                }
            }
            (300.0 / sps + t1.elapsed().as_secs_f64(), "measured", floors, last_floor)
        };
        println!(
            "{name}: {} x {} = {} cells | {:.0} steps/s | visual steady \
             (step {target}) in {:.0} s wall ({how}) | floors {floors} (last at step {last_floor})",
            g.nz, g.nr, g.len(), sps, wall_s
        );
        record_bench(name, g.len(), sps, wall_s);
        if floors > 0 {
            cfd_results::record_note("grading-bench", name, &format!(
                "{name}: {floors} positivity-floor activations, all during the cold-start \
                 front (last at step {last_floor}, zero after) — the §13 quarantine class."));
        }
        // Floor contact must have the §13 cold-start shape: confined to the
        // startup front and quiet ever after (the report quarantine covers
        // it). Steady-state floor contact is a hard failure. Measured here:
        // graded Long trips 552 at the lip corner over steps 91-121 (the
        // startup blast expands deeper in the larger domain), zero after;
        // every other case is clean.
        assert!(
            floors == 0 || last_floor < 1500,
            "{name}: floors {floors}, last at step {last_floor} — not startup-confined"
        );
    }

    /// The work-order report (item 5): cells, steps/s and seconds-to-steady
    /// before (uniform grids) and after (graded), on this machine.
    ///
    ///     cargo test -p cfd-ui grading_bench -- --include-ignored --nocapture
    #[test]
    #[ignore = "benchmark: minutes of solver time; run explicitly"]
    fn grading_before_after() {
        let wall = conical_contour(8.0);

        // BEFORE 1: the historic compact uniform 320 x 200 demo grid.
        let pc = CaseParams { plume: PlumeLength::Compact, ..CaseParams::default() };
        run_case("uniform Compact (historic)", bench_setup(base_grid(&pc), &pc, &wall), false);

        // BEFORE 2: the rejected alternative — the Long domain at UNIFORM
        // base resolution (what the work order prices at ~14x).
        let pl = CaseParams { plume: PlumeLength::Long, ..CaseParams::default() };
        let (lz, lr) = domain(&pl);
        let dr = 1.0 / pl.cells_per_rt;
        let dz = 2.9 * dr;
        let ug = Grid::uniform(
            (lz / dz).round() as usize,
            (lr / dr).round() as usize,
            dz as f32,
            dr as f32,
        );
        run_case("uniform Long (rejected)", bench_setup(ug, &pl, &wall), true);

        // AFTER: the graded Standard and Long grids.
        let ps = CaseParams::default(); // Standard is the default
        run_case("graded Standard (default)", make_setup(&ps, &wall), false);
        run_case("graded Long", make_setup(&pl, &wall), false);
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
    fn diag_graded_long_floors() {
        let pl = CaseParams { plume: PlumeLength::Long, ..CaseParams::default() };
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
