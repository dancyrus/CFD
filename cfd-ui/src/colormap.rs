//! 256-entry colormap LUTs, one per `FieldKind`. Built once, indexed per
//! pixel — the schlieren exponential is folded into the LUT (docs/sessions/
//! D-ui.md: that is the difference between 0.5 ms and 2.1 ms per frame).

use std::sync::OnceLock;

use cfd_contract::FieldKind;

pub type Lut = [[u8; 4]; 256];

/// Wall cells are painted this, never through a LUT.
pub const SOLID_RGBA: [u8; 4] = [72, 74, 82, 255];

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

fn viridis() -> Lut {
    build(&[
        (0.000, [68, 1, 84]),
        (0.125, [71, 44, 122]),
        (0.250, [59, 81, 139]),
        (0.375, [44, 113, 142]),
        (0.500, [33, 144, 141]),
        (0.625, [39, 173, 129]),
        (0.750, [92, 200, 99]),
        (0.875, [170, 220, 50]),
        (1.000, [253, 231, 37]),
    ])
}

fn inferno() -> Lut {
    build(&[
        (0.000, [0, 0, 4]),
        (0.125, [31, 12, 72]),
        (0.250, [85, 15, 109]),
        (0.375, [136, 34, 106]),
        (0.500, [186, 54, 85]),
        (0.625, [227, 89, 51]),
        (0.750, [249, 140, 10]),
        (0.875, [249, 201, 50]),
        (1.000, [252, 255, 164]),
    ])
}

fn turbo() -> Lut {
    build(&[
        (0.000, [48, 18, 59]),
        (0.125, [70, 107, 227]),
        (0.250, [40, 178, 251]),
        (0.375, [27, 229, 181]),
        (0.500, [97, 252, 108]),
        (0.625, [180, 241, 52]),
        (0.750, [246, 191, 26]),
        (0.875, [249, 105, 8]),
        (1.000, [122, 4, 3]),
    ])
}

/// Diverging blue-white-red for signed fields (velocities).
fn coolwarm() -> Lut {
    build(&[
        (0.00, [59, 76, 192]),
        (0.25, [124, 159, 249]),
        (0.50, [222, 220, 218]),
        (0.75, [245, 152, 105]),
        (1.00, [180, 4, 38]),
    ])
}

/// Schlieren: dark features on a light ground, with the exponential response
/// exp(-k·x) folded into the index so the per-pixel path stays linear.
fn schlieren() -> Lut {
    let mut lut = [[0, 0, 0, 255]; 256];
    for (k, e) in lut.iter_mut().enumerate() {
        let x = k as f32 / 255.0;
        let shade = (245.0 * (-5.0 * x).exp()) as u8;
        *e = [shade, shade, shade, 255];
    }
    lut
}

pub fn lut_for(kind: FieldKind) -> &'static Lut {
    static LUTS: OnceLock<[Lut; 8]> = OnceLock::new();
    let luts = LUTS.get_or_init(|| {
        let mut l = [[[0u8; 4]; 256]; 8];
        l[FieldKind::Density as usize] = viridis();
        l[FieldKind::Pressure as usize] = viridis();
        l[FieldKind::Temperature as usize] = inferno();
        l[FieldKind::Mach as usize] = turbo();
        l[FieldKind::VelocityZ as usize] = coolwarm();
        l[FieldKind::VelocityR as usize] = coolwarm();
        l[FieldKind::Speed as usize] = turbo();
        l[FieldKind::Schlieren as usize] = schlieren();
        l
    });
    &luts[kind as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luts_cover_endpoints() {
        for k in FieldKind::ALL {
            let lut = lut_for(k);
            assert_eq!(lut.len(), 256);
            assert!(lut.iter().all(|c| c[3] == 255));
        }
        // Schlieren is monotone decreasing (dark = strong gradient).
        let s = lut_for(FieldKind::Schlieren);
        assert!(s[0][0] > s[128][0] && s[128][0] > s[255][0]);
    }
}
