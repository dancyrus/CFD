//! 256-entry colormap LUTs. Colormap identity is a [`Preset`] the player picks
//! per field, not a property of the field — `docs/colormap-style-guide.md` §2
//! gives the per-field defaults, §3 the control points.
//!
//! Anchor positions are deliberately NOT evenly spaced. Each table is the
//! minimal anchor set reproducing its reference map to max ΔE00 ≤ 2.0 under the
//! piecewise-linear-in-sRGB `build()` below, emitted by
//! `docs/tools/verify_colormap.py`. Resampling them to even spacing is the
//! defect §4 documents: it is what put the old 9-point turbo 21.6 ΔE00 away
//! from real turbo and gave the old 5-point coolwarm a Mach band at zero.
//! Regenerate with the tool rather than hand-editing.

use std::sync::OnceLock;

use cfd_contract::FieldKind;

pub type Lut = [[u8; 4]; 256];

/// Wall cells are painted this, never through a LUT.
pub const SOLID_RGBA: [u8; 4] = [72, 74, 82, 255];

/// How a preset spends its lightness (style guide §1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PresetKind {
    /// Lightness monotone end to end.
    Sequential,
    /// Light apex in the middle. Only says something true when the range is
    /// centred on a value that means something (§8).
    Diverging,
    /// No hue at all.
    Neutral,
}

/// A colormap. Decoupled from `FieldKind`: any preset can be selected for any
/// field, with the guidance carried by warnings rather than by filtering (§2).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Preset {
    Viridis = 0,
    Inferno = 1,
    BlackBody = 2,
    Batlow = 3,
    Vik = 4,
    SmoothCoolWarm = 5,
    Grayscale = 6,
    Turbo = 7,
}

impl Preset {
    /// Picker order. Turbo sits last: it is offered, but it is not a default
    /// for anything (§7).
    pub const ALL: [Preset; 8] = [
        Preset::Viridis,
        Preset::Inferno,
        Preset::BlackBody,
        Preset::Batlow,
        Preset::Vik,
        Preset::SmoothCoolWarm,
        Preset::Grayscale,
        Preset::Turbo,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Preset::Viridis => "viridis",
            Preset::Inferno => "inferno",
            Preset::BlackBody => "black-body",
            Preset::Batlow => "batlow",
            Preset::Vik => "vik",
            Preset::SmoothCoolWarm => "smooth-cool-warm",
            Preset::Grayscale => "grayscale",
            // §7: an engineered rainbow, offered for players trained on Fluent
            // and EnSight. Labelled so it reads as the legacy choice it is.
            Preset::Turbo => "Turbo (classic)",
        }
    }

    pub fn kind(self) -> PresetKind {
        match self {
            Preset::Viridis
            | Preset::Inferno
            | Preset::BlackBody
            | Preset::Batlow
            | Preset::Turbo => PresetKind::Sequential,
            Preset::Vik | Preset::SmoothCoolWarm => PresetKind::Diverging,
            Preset::Grayscale => PresetKind::Neutral,
        }
    }

    /// The §2 default assignment. Defaults only — the picker can override any
    /// of these, and the choice persists per field.
    pub fn default_for(field: FieldKind) -> Preset {
        match field {
            FieldKind::Density => Preset::Viridis,
            FieldKind::Pressure => Preset::Inferno,
            FieldKind::Temperature => Preset::BlackBody,
            FieldKind::Mach => Preset::Batlow,
            FieldKind::VelocityZ | FieldKind::VelocityR => Preset::Vik,
            FieldKind::Speed => Preset::Inferno,
            FieldKind::Schlieren => Preset::Grayscale,
        }
    }

    /// Control points, or `None` for a preset built procedurally.
    ///
    /// Positions are copied verbatim from style guide §3. Do not round them and
    /// do not respace them (§4).
    fn anchors(self) -> Option<&'static [(f32, [u8; 3])]> {
        Some(match self {
            // 7 anchors · max ΔE00 1.50 · L* 14.9→90.9 · step CV 1.2%
            Preset::Viridis => &[
                (0.0000, [68, 1, 84]),
                (0.1725, [68, 59, 132]),
                (0.4157, [40, 124, 142]),
                (0.5647, [31, 160, 136]),
                (0.7059, [70, 192, 111]),
                (0.8784, [173, 220, 48]),
                (1.0000, [253, 231, 37]),
            ],
            // 11 anchors · max ΔE00 1.96 · L* 0.1→97.9 · step CV 1.1%
            Preset::Inferno => &[
                (0.0000, [0, 0, 4]),
                (0.0627, [11, 7, 36]),
                (0.1647, [50, 10, 94]),
                (0.2824, [100, 21, 110]),
                (0.4314, [160, 42, 99]),
                (0.5961, [219, 80, 59]),
                (0.7294, [247, 132, 16]),
                (0.8235, [252, 176, 20]),
                (0.9059, [245, 217, 73]),
                (0.9569, [241, 241, 121]),
                (1.0000, [252, 255, 164]),
            ],
            // 7 anchors · max ΔE00 1.94 · L* 0.0→100.0 · linear in L*.
            // The 59% step CV is accepted, not a defect: this map is designed
            // for linear L*, not constant ΔE00, and the spikes sit at the
            // near-black end where 8-bit quantisation dominates (§3).
            Preset::BlackBody => &[
                (0.0000, [0, 0, 0]),
                (0.0235, [18, 6, 3]),
                (0.0745, [38, 16, 10]),
                (0.3922, [178, 34, 34]),
                (0.5843, [227, 105, 5]),
                (0.8863, [230, 229, 53]),
                (1.0000, [255, 255, 255]),
            ],
            // 7 anchors · max ΔE00 1.80 · L* 12.2→87.2 · step CV 8.9%
            Preset::Batlow => &[
                (0.0000, [1, 25, 89]),
                (0.1137, [16, 64, 96]),
                (0.2745, [40, 100, 95]),
                (0.5569, [157, 137, 43]),
                (0.6627, [209, 147, 66]),
                (0.7608, [244, 159, 114]),
                (1.0000, [250, 204, 250]),
            ],
            // 9 anchors · max ΔE00 1.27 · L* 11.2→91.6 (apex at 0.49)→16.3.
            // The anchor at 0.4902 is the light apex and is not decoration:
            // drop it and the lightness derivative flips sign discontinuously,
            // painting a shear layer at u = 0 that is not there (§3).
            Preset::Vik => &[
                (0.0000, [0, 18, 97]),
                (0.1686, [6, 86, 140]),
                (0.2549, [51, 127, 168]),
                (0.4314, [192, 216, 228]),
                (0.4902, [232, 231, 229]),
                (0.5490, [236, 209, 195]),
                (0.8118, [179, 83, 31]),
                (0.8824, [143, 43, 6]),
                (1.0000, [89, 0, 8]),
            ],
            // 10 anchors · max ΔE00 1.11 · step CV 19.2%. ParaView's classic
            // blue-white-red, kept for players who expect it (§3).
            Preset::SmoothCoolWarm => &[
                (0.0000, [59, 76, 192]),
                (0.1412, [103, 136, 238]),
                (0.2706, [148, 182, 255]),
                (0.3922, [190, 211, 246]),
                (0.5020, [221, 220, 220]),
                (0.6314, [245, 195, 170]),
                (0.7686, [242, 146, 116]),
                (0.8980, [215, 84, 69]),
                (0.9529, [198, 52, 52]),
                (1.0000, [180, 4, 38]),
            ],
            // 13 anchors · max ΔE00 1.79 vs real turbo. This is the table §7
            // calls for; the 9-point resample it replaces sat 21.6 ΔE00 away.
            // Emitted by `python docs/tools/verify_colormap.py turbo`.
            Preset::Turbo => &[
                (0.0000, [48, 18, 59]),
                (0.0902, [68, 84, 195]),
                (0.1451, [71, 120, 240]),
                (0.1922, [65, 150, 255]),
                (0.2549, [39, 190, 233]),
                (0.3412, [29, 231, 178]),
                (0.4667, [136, 255, 78]),
                (0.5412, [190, 244, 52]),
                (0.6275, [238, 207, 58]),
                (0.6941, [254, 169, 51]),
                (0.7647, [249, 117, 29]),
                (0.8745, [210, 49, 5]),
                (1.0000, [122, 4, 3]),
            ],
            // Procedural: see `grayscale()`.
            Preset::Grayscale => return None,
        })
    }
}

fn build(anchors: &[(f32, [u8; 3])]) -> Lut {
    let mut lut = [[0, 0, 0, 255]; 256];
    for (k, e) in lut.iter_mut().enumerate() {
        let x = k as f32 / 255.0;
        let seg = anchors.windows(2).find(|w| x >= w[0].0 && x <= w[1].0);
        let (a, b) = match seg {
            Some(w) => (w[0], w[1]),
            None if x < anchors[0].0 => (anchors[0], anchors[0]),
            None => (anchors[anchors.len() - 1], anchors[anchors.len() - 1]),
        };
        let t = if b.0 > a.0 {
            (x - a.0) / (b.0 - a.0)
        } else {
            0.0
        };
        for (c, ch) in e.iter_mut().take(3).enumerate() {
            *ch = (a.1[c] as f32 + t * (b.1[c] as f32 - a.1[c] as f32)).round() as u8;
        }
    }
    lut
}

/// Schlieren: dark features on a light ground, with the exponential response
/// exp(-k·x) folded into the LUT values so the per-pixel path stays one index.
///
/// This still collapses the strong-gradient end into few distinct greys — the
/// §5 work order moves `exp(-k·S)` into the `cfd-core` snapshot pass and leaves
/// a plain grey ramp here. That crosses a crate boundary and is out of scope for
/// this change; §5's stated interim applies instead — round rather than
/// truncate, which was biasing every level down by up to one code value.
fn grayscale() -> Lut {
    let mut lut = [[0, 0, 0, 255]; 256];
    for (k, e) in lut.iter_mut().enumerate() {
        let x = k as f32 / 255.0;
        let shade = (245.0 * (-5.0 * x).exp()).round() as u8;
        *e = [shade, shade, shade, 255];
    }
    lut
}

pub fn lut_for(preset: Preset) -> &'static Lut {
    static LUTS: OnceLock<[Lut; 8]> = OnceLock::new();
    let luts = LUTS.get_or_init(|| {
        let mut l = [[[0u8; 4]; 256]; 8];
        for p in Preset::ALL {
            l[p as usize] = match p.anchors() {
                Some(a) => build(a),
                None => grayscale(),
            };
        }
        l
    });
    &luts[preset as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luts_cover_endpoints() {
        for p in Preset::ALL {
            let lut = lut_for(p);
            assert_eq!(lut.len(), 256, "{}", p.name());
            assert!(
                lut.iter().all(|c| c[3] == 255),
                "{} has a non-opaque entry",
                p.name()
            );
            let Some(a) = p.anchors() else { continue };
            // Anchors must span the whole unit interval, or `build()` clamps
            // the ends flat and the map silently loses range.
            assert_eq!(a[0].0, 0.0, "{} does not start at 0.0", p.name());
            assert_eq!(a[a.len() - 1].0, 1.0, "{} does not end at 1.0", p.name());
            assert!(
                a.windows(2).all(|w| w[1].0 > w[0].0),
                "{} anchor positions are not strictly increasing",
                p.name()
            );
        }
        // Grayscale/schlieren is monotone decreasing (dark = strong gradient).
        let s = lut_for(Preset::Grayscale);
        assert!(s[0][0] > s[128][0] && s[128][0] > s[255][0]);
    }

    /// §7: turbo is offered, but it is not the default for anything.
    #[test]
    fn turbo_is_no_fields_default() {
        for k in FieldKind::ALL {
            assert_ne!(
                Preset::default_for(k),
                Preset::Turbo,
                "{} defaults to turbo",
                k.label()
            );
        }
    }

    /// The §2 table, asserted so code and doc cannot drift apart silently.
    #[test]
    fn defaults_match_style_guide() {
        assert_eq!(Preset::default_for(FieldKind::Density), Preset::Viridis);
        assert_eq!(Preset::default_for(FieldKind::Pressure), Preset::Inferno);
        assert_eq!(
            Preset::default_for(FieldKind::Temperature),
            Preset::BlackBody
        );
        assert_eq!(Preset::default_for(FieldKind::Mach), Preset::Batlow);
        assert_eq!(Preset::default_for(FieldKind::VelocityZ), Preset::Vik);
        assert_eq!(Preset::default_for(FieldKind::VelocityR), Preset::Vik);
        assert_eq!(Preset::default_for(FieldKind::Speed), Preset::Inferno);
        assert_eq!(Preset::default_for(FieldKind::Schlieren), Preset::Grayscale);
    }

    /// Every signed field must default to a diverging map, and every unsigned
    /// one to something that is not — the §2 rationale in executable form.
    #[test]
    fn signed_fields_default_to_diverging() {
        for k in FieldKind::ALL {
            let diverging = Preset::default_for(k).kind() == PresetKind::Diverging;
            assert_eq!(diverging, k.is_signed(), "{}", k.label());
        }
    }
}
