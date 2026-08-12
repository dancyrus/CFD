//! Pure, stateless flux and reconstruction kernels. **Frozen.**
//!
//! These live here, not in the solver crate, so both solver sessions share a
//! compiled artifact rather than an agreement about behaviour. Contracts by
//! signature drift at merge; contracts by implementation cannot.
//!
//! Everything operates on `Prim = [rho, u_n, u_t, p]` rotated so the face
//! normal is +n. `hllc_flux`/`hll_flux` return a `Cons` in that same rotated
//! frame: `[mass, n-momentum, t-momentum, energy]`. The caller rotates back.

use crate::{Cons, Limiter, Prim, Real, Reconstruction};

/// Zero when the arguments disagree in sign, else the smaller magnitude.
pub fn minmod(a: Real, b: Real) -> Real {
    if a * b <= 0.0 { 0.0 } else if a.abs() < b.abs() { a } else { b }
}

/// Canonical [rho, u_z, u_r, p] from conserved [rho, rho*u_z, rho*u_r, E].
pub fn cons_to_prim(u: Cons, gamma: Real) -> Prim {
    let rho = u[0];
    let inv = 1.0 / rho;
    let uz = u[1] * inv;
    let ur = u[2] * inv;
    let p = (gamma - 1.0) * (u[3] - 0.5 * rho * (uz * uz + ur * ur));
    [rho, uz, ur, p]
}

pub fn prim_to_cons(w: Prim, gamma: Real) -> Cons {
    let [rho, un, ut, p] = w;
    [rho, rho * un, rho * ut, p / (gamma - 1.0) + 0.5 * rho * (un * un + ut * ut)]
}

pub fn sound_speed(w: Prim, gamma: Real) -> Real {
    (gamma * w[3] / w[0]).sqrt()
}

/// Analytic flux of a primitive state in the rotated frame.
#[inline]
fn analytic_flux(w: Prim, gamma: Real) -> Cons {
    let [rho, un, ut, p] = w;
    let e = p / (gamma - 1.0) + 0.5 * rho * (un * un + ut * ut);
    [rho * un, rho * un * un + p, rho * un * ut, un * (e + p)]
}

/// Toro's S_M formulation with Davis wave-speed estimates. Direction-agnostic:
/// caller rotates so the face normal is +n. Returns Cons in the SAME rotated
/// frame. Resolves an isolated contact exactly — that property is what keeps
/// the plume boundary sharp, and it is what the Sod-star-state test asserts.
pub fn hllc_flux(ql: Prim, qr: Prim, gamma: Real) -> Cons {
    let [rl, ul, vl, pl] = ql;
    let [rr, ur, vr, pr] = qr;
    let al = sound_speed(ql, gamma);
    let ar = sound_speed(qr, gamma);

    // Davis estimates.
    let sl = (ul - al).min(ur - ar);
    let sr = (ul + al).max(ur + ar);

    if sl >= 0.0 { return analytic_flux(ql, gamma); }
    if sr <= 0.0 { return analytic_flux(qr, gamma); }

    // Contact/star speed, Toro (10.37).
    let ml = rl * (sl - ul); // rho_L * (S_L - u_L), negative
    let mr = rr * (sr - ur); // rho_R * (S_R - u_R), positive
    let sm = (pr - pl + ul * ml - ur * mr) / (ml - mr);

    if sm >= 0.0 {
        let el = pl / (gamma - 1.0) + 0.5 * rl * (ul * ul + vl * vl);
        let rs = ml / (sl - sm); // star density
        let es = rs * (el / rl + (sm - ul) * (sm + pl / ml));
        let us = [rs, rs * sm, rs * vl, es];
        let ul_c = [rl, rl * ul, rl * vl, el];
        let f = analytic_flux(ql, gamma);
        [f[0] + sl * (us[0] - ul_c[0]), f[1] + sl * (us[1] - ul_c[1]),
         f[2] + sl * (us[2] - ul_c[2]), f[3] + sl * (us[3] - ul_c[3])]
    } else {
        let er = pr / (gamma - 1.0) + 0.5 * rr * (ur * ur + vr * vr);
        let rs = mr / (sr - sm);
        let es = rs * (er / rr + (sm - ur) * (sm + pr / mr));
        let us = [rs, rs * sm, rs * vr, es];
        let ur_c = [rr, rr * ur, rr * vr, er];
        let f = analytic_flux(qr, gamma);
        [f[0] + sr * (us[0] - ur_c[0]), f[1] + sr * (us[1] - ur_c[1]),
         f[2] + sr * (us[2] - ur_c[2]), f[3] + sr * (us[3] - ur_c[3])]
    }
}

/// Same contract as `hllc_flux`. Used where the carbuncle sensor fires.
pub fn hll_flux(ql: Prim, qr: Prim, gamma: Real) -> Cons {
    let al = sound_speed(ql, gamma);
    let ar = sound_speed(qr, gamma);
    let sl = (ql[1] - al).min(qr[1] - ar);
    let sr = (ql[1] + al).max(qr[1] + ar);

    if sl >= 0.0 { return analytic_flux(ql, gamma); }
    if sr <= 0.0 { return analytic_flux(qr, gamma); }

    let ul = prim_to_cons(ql, gamma);
    let ur = prim_to_cons(qr, gamma);
    let fl = analytic_flux(ql, gamma);
    let fr = analytic_flux(qr, gamma);
    let inv = 1.0 / (sr - sl);
    [(sr * fl[0] - sl * fr[0] + sl * sr * (ur[0] - ul[0])) * inv,
     (sr * fl[1] - sl * fr[1] + sl * sr * (ur[1] - ul[1])) * inv,
     (sr * fl[2] - sl * fr[2] + sl * sr * (ur[2] - ul[2])) * inv,
     (sr * fl[3] - sl * fr[3] + sl * sr * (ur[3] - ul[3])) * inv]
}

#[inline]
fn slope(dm: Real, dp: Real, lim: Limiter) -> Real {
    match lim {
        Limiter::None => 0.5 * (dm + dp),
        Limiter::Minmod => minmod(dm, dp),
        Limiter::VanLeer => {
            if dm * dp > 0.0 { 2.0 * dm * dp / (dm + dp) } else { 0.0 }
        }
    }
}

/// s = four consecutive cells straddling face i+1/2: [i-1, i, i+1, i+2].
/// Any true in `solid` drops that side to first order. Returns (left, right)
/// face states. Reconstruction::FirstOrder returns the cell averages unchanged.
///
/// A fluid cell touching a wall becomes piecewise-constant in the wall-normal
/// direction — that is the entire stencil degradation. See
/// docs/physics-reference.md §3.
pub fn muscl_face_states(
    s: [Prim; 4], solid: [bool; 4], recon: Reconstruction, lim: Limiter,
) -> (Prim, Prim) {
    if recon == Reconstruction::FirstOrder {
        return (s[1], s[2]);
    }
    let mut left = s[1];
    let mut right = s[2];
    let l_solid = solid[0] || solid[2];
    let r_solid = solid[1] || solid[3];
    for k in 0..4 {
        if !l_solid {
            left[k] += 0.5 * slope(s[1][k] - s[0][k], s[2][k] - s[1][k], lim);
        }
        if !r_solid {
            right[k] -= 0.5 * slope(s[2][k] - s[1][k], s[3][k] - s[2][k], lim);
        }
    }
    (left, right)
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact Riemann solver star state (f64, Newton on the pressure function).
    /// Test-only: validates the literals the HLLC test uses.
    fn exact_star(rl: f64, ul: f64, pl: f64, rr: f64, ur: f64, pr: f64, g: f64) -> (f64, f64) {
        let al = (g * pl / rl).sqrt();
        let ar = (g * pr / rr).sqrt();
        let f = |p: f64, rk: f64, pk: f64, ak: f64| -> (f64, f64) {
            if p > pk {
                // shock
                let a_k = 2.0 / ((g + 1.0) * rk);
                let b_k = (g - 1.0) / (g + 1.0) * pk;
                let q = (a_k / (p + b_k)).sqrt();
                ((p - pk) * q, q * (1.0 - (p - pk) / (2.0 * (p + b_k))))
            } else {
                // rarefaction
                let pr_ = p / pk;
                (2.0 * ak / (g - 1.0) * (pr_.powf((g - 1.0) / (2.0 * g)) - 1.0),
                 1.0 / (rk * ak) * pr_.powf(-(g + 1.0) / (2.0 * g)))
            }
        };
        let mut p = 0.5 * (pl + pr);
        for _ in 0..60 {
            let (fl, dl) = f(p, rl, pl, al);
            let (fr, dr) = f(p, rr, pr, ar);
            let step = (fl + fr + (ur - ul)) / (dl + dr);
            p -= step;
            if step.abs() < 1e-14 * p { break; }
        }
        let (fl, _) = f(p, rl, pl, al);
        let (fr, _) = f(p, rr, pr, ar);
        let u = 0.5 * (ul + ur) + 0.5 * (fr - fl);
        (p, u)
    }

    // The Sod star state, independently recomputed. docs/physics-reference.md §12.
    const P_STAR: f64 = 0.3031301781;
    const U_STAR: f64 = 0.9274526200;
    const RHO_STAR_L: f64 = 0.4263194282;
    const RHO_STAR_R: f64 = 0.2655737117;

    #[test]
    fn exact_riemann_reproduces_sod_reference() {
        let (p, u) = exact_star(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4);
        assert!((p - P_STAR).abs() < 1e-9, "p* = {p}");
        assert!((u - U_STAR).abs() < 1e-9, "u* = {u}");
        // Star densities: isentropic on the left, Rankine-Hugoniot on the right.
        let rsl = 1.0 * (p / 1.0_f64).powf(1.0 / 1.4);
        let gr = (1.4 - 1.0) / (1.4 + 1.0);
        let rsr = 0.125 * ((p / 0.1 + gr) / (gr * p / 0.1 + 1.0));
        assert!((rsl - RHO_STAR_L).abs() < 1e-9, "rho*_L = {rsl}");
        assert!((rsr - RHO_STAR_R).abs() < 1e-9, "rho*_R = {rsr}");
    }

    /// hllc_flux across the Sod interface — the contact discontinuity between
    /// the two exact star states — reproduces the exact star state: HLLC's
    /// contact resolution must return exactly the upwind analytic flux
    /// rho*_L·u*, rho*_L·u*² + p*, u*(E*+p*).
    #[test]
    fn hllc_resolves_sod_star_contact_exactly() {
        let g: Real = 1.4;
        let ql: Prim = [RHO_STAR_L as Real, U_STAR as Real, 0.0, P_STAR as Real];
        let qr: Prim = [RHO_STAR_R as Real, U_STAR as Real, 0.0, P_STAR as Real];
        let f = hllc_flux(ql, qr, g);
        // u* > 0, so the exact interface flux is the analytic flux of the
        // left star state.
        let e = P_STAR / 0.4 + 0.5 * RHO_STAR_L * U_STAR * U_STAR;
        let expect = [RHO_STAR_L * U_STAR,
                      RHO_STAR_L * U_STAR * U_STAR + P_STAR,
                      0.0,
                      U_STAR * (e + P_STAR)];
        for k in 0..4 {
            assert!((f[k] as f64 - expect[k]).abs() <= 1e-5 * expect[k].abs().max(1.0),
                    "component {k}: got {}, want {}", f[k], expect[k]);
        }
    }

    #[test]
    fn hllc_and_hll_are_consistent_on_uniform_states() {
        let g: Real = 1.4;
        for q in [[1.0, 0.0, 0.0, 1.0], [0.7, 2.0, -0.3, 0.4], [2.0, -1.5, 0.2, 3.0]] {
            let fa = analytic_flux(q, g);
            for f in [hllc_flux(q, q, g), hll_flux(q, q, g)] {
                for k in 0..4 {
                    assert!((f[k] - fa[k]).abs() <= 1e-6 * fa[k].abs().max(1.0));
                }
            }
        }
    }

    #[test]
    fn hllc_upwinds_supersonic_states() {
        let g: Real = 1.4;
        let ql: Prim = [1.0, 3.0, 0.1, 1.0]; // M ~ 2.5 rightward
        let qr: Prim = [0.5, 3.0, 0.0, 0.5];
        let f = hllc_flux(ql, qr, g);
        let fa = analytic_flux(ql, g);
        for k in 0..4 { assert_eq!(f[k], fa[k]); }
    }

    #[test]
    fn muscl_reproduces_linear_field_exactly() {
        // d = 0.25 is exactly representable, so reconstruction must be exact.
        for lim in [Limiter::Minmod, Limiter::VanLeer, Limiter::None] {
            let mut s = [[0.0 as Real; 4]; 4];
            for (i, c) in s.iter_mut().enumerate() {
                for k in 0..4 { c[k] = 1.0 + 0.25 * i as Real + 0.5 * k as Real; }
            }
            let (l, r) = muscl_face_states(s, [false; 4], Reconstruction::Muscl, lim);
            for k in 0..4 {
                let face = 0.5 * (s[1][k] + s[2][k]);
                assert_eq!(l[k], face, "{lim:?} left k={k}");
                assert_eq!(r[k], face, "{lim:?} right k={k}");
            }
        }
    }

    #[test]
    fn muscl_degrades_to_first_order_at_walls() {
        let s = [[1.0, 0.0, 0.0, 1.0], [2.0, 1.0, 0.0, 2.0],
                 [3.0, 2.0, 0.0, 3.0], [4.0, 3.0, 0.0, 4.0]];
        // Solid at i-1: the left state must be the bare cell average.
        let (l, _) = muscl_face_states(s, [true, false, false, false],
                                       Reconstruction::Muscl, Limiter::Minmod);
        assert_eq!(l, s[1]);
        // Solid at i+2: the right state must be the bare cell average.
        let (_, r) = muscl_face_states(s, [false, false, false, true],
                                       Reconstruction::Muscl, Limiter::Minmod);
        assert_eq!(r, s[2]);
        // FirstOrder returns cell averages regardless.
        let (l, r) = muscl_face_states(s, [false; 4], Reconstruction::FirstOrder,
                                       Limiter::Minmod);
        assert_eq!(l, s[1]);
        assert_eq!(r, s[2]);
    }

    #[test]
    fn prim_cons_round_trip_to_1e6_relative() {
        let g: Real = 1.4;
        let states: [Cons; 4] = [
            [1.0, 0.0, 0.0, 2.5],
            [0.125, 0.0, 0.0, 0.25],
            [2.0, 4.0, -1.0, 20.0],
            [1e-4, 1e-5, 2e-5, 3e-4],
        ];
        for u in states {
            let v = prim_to_cons(cons_to_prim(u, g), g);
            for k in 0..4 {
                let scale = u[k].abs().max(1e-30);
                assert!((v[k] - u[k]).abs() / scale <= 1e-6, "{u:?} -> {v:?}");
            }
        }
    }

    #[test]
    fn minmod_basics() {
        assert_eq!(minmod(1.0, 2.0), 1.0);
        assert_eq!(minmod(-2.0, -1.0), -1.0);
        assert_eq!(minmod(1.0, -1.0), 0.0);
        assert_eq!(minmod(0.0, 5.0), 0.0);
    }
}
