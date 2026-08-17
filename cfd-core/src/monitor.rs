//! Convergence monitor — a **plateau on the trailing MEAN** of a monitored
//! scalar, sampled at fixed **physical-time** intervals.
//!
//! Two things this deliberately is not.
//!
//! - **Not the density residual.** The old criterion, `residual < 1e-3`, never
//!   fired on the demo case: a mildly overexpanded sea-level plume breathes and
//!   has no steady state, so the normalised residual bottoms out near 8e-2 and
//!   then *rises* and oscillates. `converged` gates the report, so the user
//!   could never see a valid report on the default case. That was a
//!   correctness bug, not a slow finish line. The residual survives as a
//!   reported diagnostic (`StepInfo::residual`) and nothing else.
//! - **Not a peak-to-peak spread over the window.** Peak-to-peak has to wait
//!   out the breathing *amplitude*; the trailing mean is stationary as soon as
//!   the transient decays. Both were measured on the same run — see
//!   `docs/work-orders/convergence-plateau.md` and
//!   `cfd-core/tests/convergence.rs`.
//!
//! Sampling is by physical time, never by step count. `dt` is CFL-limited by a
//! field that changes, so it varies over a run; and under the local time
//! stepping sketched in the physics reference, step count stops meaning
//! anything at all. The window length is auto-sized from the domain transit
//! time `L / (|u| + a)_ref`, so it follows the domain the user actually chose
//! instead of a hardcoded step count that is only right for one mesh.
//!
//! The monitor is signal-agnostic; `step.rs` feeds it the **report's
//! exit-plane thrust** — the headline number the verdict un-greys, and the
//! most plume-sensitive of the report quantities, so when its mean has
//! stopped moving the rest of the report has too. Two candidates measured
//! and rejected on the demo case (survey recorded in
//! `docs/results/convergence-<machine>.json`): the domain-integrated axial
//! momentum tracks the FULL acoustic equilibration of the far field
//! (~15,800 steps, §9) rather than the report, and the mass flow at the lip
//! shows isolated lucky-dip windows well before its mean actually settles.

/// Samples per window. The window's *physical duration* is what the transit
/// time sets; this only says how finely that duration is sampled.
pub const SAMPLES_PER_WINDOW: usize = 16;

/// Window duration as a fraction of one domain transit, `L / (|u| + a)_ref`.
/// A verdict needs two full windows, so the earliest possible fire is one
/// whole transit — the startup front cannot outrun it.
pub const WINDOW_TRANSITS: f64 = 0.5;

/// Relative tolerance on the trailing mean. The plateau test itself uses
/// `tol/2`: the criterion is that the window mean has moved less than half a
/// tolerance from the previous window's mean, so two consecutive verdicts
/// bracket a full `tol` of drift.
///
/// 2e-2 means "the mean thrust moved less than 1% across half a domain
/// transit". Sized from two measurements, not taste: (a) the thrust window
/// mean's drift on the settled demo case oscillates in a ~1e-3..8e-3 band
/// forever (undamped plume breathing), so a threshold below ~1e-2 either
/// never fires or fires on a lucky low beat of that oscillation; (b) the
/// number the verdict un-greys carries ±1/N_throat mass-flow quantization
/// (±5% at the default 20 cells/r_t) and the §12/T8 staircase thrust bias
/// (~13% at that resolution), so a 1%-per-window drift bound is 5-13x
/// tighter than the accuracy of the quantity it protects.
pub const CONVERGED_TOL: f64 = 2e-2;

/// Reference signal speed `(|u| + a)_ref`, non-dimensional (chamber-referenced,
/// so `p0 = rho0 = 1`, `R = 1`, `a0² = gamma`).
///
/// This is the **largest** `|u| + a` any fluid particle isentropically expanded
/// from the chamber can carry: maximising `u + a` under
/// `u²/2 + a²/(gamma-1) = h0 = gamma/(gamma-1)` gives
/// `a² = gamma(gamma-1)/(gamma+1)` and `u = 2a/(gamma-1)`, hence
/// `u + a = sqrt(gamma (gamma+1) / (gamma-1))`.
///
/// Using the fastest attainable signal makes the transit time — and therefore
/// the window — the *shortest* defensible one, which is the conservative
/// direction for a convergence test: a shorter window fires no earlier, it
/// just resolves the plateau more finely.
pub fn reference_speed(gamma: f64) -> f64 {
    (gamma * (gamma + 1.0) / (gamma - 1.0)).sqrt()
}

/// Physical (non-dimensional) duration of one plateau window for a domain of
/// axial length `lz` in a gas of ratio `gamma`.
pub fn window_time_for(lz: f64, gamma: f64) -> f64 {
    WINDOW_TRANSITS * lz / reference_speed(gamma)
}

/// Plateau monitor over one scalar signal.
///
/// Feed it `(t, x)` after every step. It samples `x` whenever the physical
/// time has advanced by `sample_dt`, averages `SAMPLES_PER_WINDOW` samples into
/// a window mean, and fires when a window mean lands within `tol/2` (relative)
/// of the previous window's mean.
///
/// The verdict **latches** until [`reset`](Self::reset). Everything that
/// moves the solution to a different problem — a new ambient, new geometry,
/// a fresh initial field — re-arms the monitor explicitly, so a latched
/// verdict always refers to an unbroken stretch of the same problem. Without
/// latching the report would flicker in and out on a borderline window while
/// the field is doing nothing at all.
#[derive(Debug, Clone)]
pub struct PlateauMonitor {
    window_time: f64,
    sample_dt: f64,
    tol: f64,
    next_sample: f64,
    sum: f64,
    count: usize,
    prev_mean: Option<f64>,
    last_mean: Option<f64>,
    last_drift: f64,
    windows: u64,
    converged: bool,
}

impl PlateauMonitor {
    /// Explicit window length, in non-dimensional time.
    pub fn new(window_time: f64, tol: f64) -> Self {
        let window_time = if window_time.is_finite() && window_time > 0.0 {
            window_time
        } else {
            f64::INFINITY // never samples, never fires: a degenerate domain cannot converge
        };
        let sample_dt = window_time / SAMPLES_PER_WINDOW as f64;
        PlateauMonitor {
            window_time,
            sample_dt,
            tol,
            // First sample one interval after t = 0, same anchor `reset` uses.
            next_sample: sample_dt,
            sum: 0.0,
            count: 0,
            prev_mean: None,
            last_mean: None,
            last_drift: f64::NAN,
            windows: 0,
            converged: false,
        }
    }

    /// Auto-sized from the domain transit time `lz / (|u| + a)_ref`.
    pub fn for_domain(lz: f64, gamma: f64, tol: f64) -> Self {
        Self::new(window_time_for(lz, gamma), tol)
    }

    /// Re-arm at physical time `t`. Drops every accumulated sample and the
    /// latched verdict: whatever comes next is a different problem.
    pub fn reset(&mut self, t: f64) {
        self.next_sample = t + self.sample_dt;
        self.sum = 0.0;
        self.count = 0;
        self.prev_mean = None;
        self.last_mean = None;
        self.last_drift = f64::NAN;
        self.windows = 0;
        self.converged = false;
    }

    /// True when the clock wants a sample at physical time `t`. The signal
    /// reduction costs a field pass, so the caller checks this first and
    /// computes the signal only when it will actually be consumed — at the
    /// default window that is roughly one pass per hundred steps.
    pub fn due(&self, t: f64) -> bool { t >= self.next_sample }

    /// Offer the signal value `x` at physical time `t`. Returns the (latched)
    /// verdict. Call once per step; it samples only when the clock says so.
    pub fn update(&mut self, t: f64, x: f64) -> bool {
        if !self.due(t) {
            return self.converged;
        }
        // A non-finite signal means the state is not a state. Throw the
        // partial window and the comparison basis away rather than poison the
        // mean; the monitor rebuilds from the next good sample.
        if !x.is_finite() {
            self.sum = 0.0;
            self.count = 0;
            self.prev_mean = None;
            self.next_sample = t + self.sample_dt;
            return self.converged;
        }
        self.sum += x;
        self.count += 1;
        self.next_sample += self.sample_dt;
        // dt coarser than the sample interval: never emit two samples for one
        // step, just re-anchor the clock.
        if self.next_sample <= t {
            self.next_sample = t + self.sample_dt;
        }
        if self.count == SAMPLES_PER_WINDOW {
            let mean = self.sum / SAMPLES_PER_WINDOW as f64;
            self.sum = 0.0;
            self.count = 0;
            self.windows += 1;
            if let Some(prev) = self.prev_mean {
                self.last_drift = relative_drift(prev, mean);
                if self.last_drift <= 0.5 * self.tol {
                    self.converged = true;
                }
            }
            self.prev_mean = Some(mean);
            self.last_mean = Some(mean);
        }
        self.converged
    }

    pub fn converged(&self) -> bool { self.converged }
    /// Physical duration of one window.
    pub fn window_time(&self) -> f64 { self.window_time }
    /// Physical time between samples.
    pub fn sample_dt(&self) -> f64 { self.sample_dt }
    /// Completed windows since the last reset.
    pub fn windows(&self) -> u64 { self.windows }
    /// Most recent window mean, or `None` before the first window closes.
    pub fn last_mean(&self) -> Option<f64> { self.last_mean }
    /// Most recent window-to-window relative drift; NaN before two windows.
    /// The criterion is `last_drift() <= tol/2`.
    pub fn last_drift(&self) -> f64 { self.last_drift }
}

/// `|b - a|` relative to the larger magnitude. Scale-free, sign-agnostic, and
/// exactly 0 for two identical values including two zeros (a signal pinned at
/// zero has plateaued; a signal crossing zero has not, and the growing
/// denominator on either side of the crossing keeps the test honest).
pub fn relative_drift(a: f64, b: f64) -> f64 {
    let d = (b - a).abs();
    let scale = a.abs().max(b.abs());
    if scale == 0.0 {
        if d == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        d / scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive a monitor with an analytic signal at a fixed step size and return
    /// the step at which it fires, if it does.
    fn fire_step(m: &mut PlateauMonitor, dt: f64, n: u64, f: impl Fn(f64) -> f64) -> Option<u64> {
        m.reset(0.0);
        let mut t = 0.0;
        for step in 1..=n {
            t += dt;
            if m.update(t, f(t)) {
                return Some(step);
            }
        }
        None
    }

    #[test]
    fn reference_speed_is_the_isentropic_maximum() {
        for &gamma in &[1.1f64, 1.2, 1.24, 1.4, 1.667] {
            let want = reference_speed(gamma);
            // Brute-force the same maximisation: u + a on u²/2 + a²/(g-1) = h0.
            let h0 = gamma / (gamma - 1.0);
            let u_max = (2.0 * h0).sqrt();
            let mut best = 0.0f64;
            for i in 0..=200_000 {
                let u = u_max * i as f64 / 200_000.0;
                let a2 = h0 * (gamma - 1.0) - 0.5 * (gamma - 1.0) * u * u;
                if a2 > 0.0 {
                    best = best.max(u + a2.sqrt());
                }
            }
            assert!((want - best).abs() <= 1e-4 * want, "gamma {gamma}: {want} vs {best}");
            // And it is strictly faster than the chamber sound speed alone.
            assert!(want > gamma.sqrt());
        }
    }

    /// A steady signal fires as soon as two windows have closed, and not one
    /// sample earlier: 2 * SAMPLES_PER_WINDOW samples at sample_dt each.
    #[test]
    fn constant_signal_fires_after_exactly_two_windows() {
        let mut m = PlateauMonitor::new(1.0, CONVERGED_TOL);
        let dt = m.sample_dt(); // one step per sample
        let fired = fire_step(&mut m, dt, 1000, |_| 3.5).unwrap();
        assert_eq!(fired, 2 * SAMPLES_PER_WINDOW as u64);
        assert_eq!(m.windows(), 2);
        assert_eq!(m.last_drift(), 0.0);
    }

    /// The sampler keys on PHYSICAL TIME, so halving dt does not change the
    /// physical time at which the verdict lands — only the step number.
    #[test]
    fn sampling_is_by_physical_time_not_step_count() {
        let mut coarse = PlateauMonitor::new(1.0, CONVERGED_TOL);
        let mut fine = PlateauMonitor::new(1.0, CONVERGED_TOL);
        let dt = coarse.sample_dt() / 4.0;
        let a = fire_step(&mut coarse, dt * 4.0, 10_000, |_| 1.0).unwrap();
        let b = fire_step(&mut fine, dt, 10_000, |_| 1.0).unwrap();
        assert_eq!(b, 4 * a, "4x the steps for the same physical time");
        assert!(((b as f64 * dt) - (a as f64 * dt * 4.0)).abs() < 1e-12);
    }

    /// dt coarser than the sample interval: one sample per step, no doubling,
    /// still converges.
    #[test]
    fn coarse_dt_takes_one_sample_per_step() {
        let mut m = PlateauMonitor::new(1.0, CONVERGED_TOL);
        let dt = 10.0 * m.sample_dt();
        let fired = fire_step(&mut m, dt, 1000, |_| 2.0).unwrap();
        assert_eq!(fired, 2 * SAMPLES_PER_WINDOW as u64);
    }

    /// The whole point of the change. A signal that has settled onto a limit
    /// cycle — stationary mean, undecaying oscillation — is converged by the
    /// trailing mean and is NEVER converged by a peak-to-peak spread. The
    /// oscillation period is deliberately not commensurate with the window.
    #[test]
    fn limit_cycle_plateaus_on_the_mean_but_never_on_peak_to_peak() {
        let breathing = |t: f64| 1.0 + 0.05 * (t * 7.3).sin();
        let mut m = PlateauMonitor::new(1.0, CONVERGED_TOL);
        let dt = m.sample_dt() / 8.0;
        assert!(fire_step(&mut m, dt, 100_000, breathing).is_some());
        // Peak-to-peak over any window is ~2 * 0.05 * mean — above the FULL
        // tolerance forever, so a spread criterion can never fire on it.
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for i in 0..100_000 {
            let v = breathing(90_000.0 + i as f64 * dt);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(hi - lo > CONVERGED_TOL, "late-time spread {}", hi - lo);
    }

    /// A signal still on its way somewhere must not fire while its
    /// per-window motion stays above tol/2: a 10%-per-window ramp is far
    /// outside the tolerance for the whole capped run.
    #[test]
    fn a_ramp_does_not_fire() {
        let mut m = PlateauMonitor::new(1.0, CONVERGED_TOL);
        let dt = m.sample_dt();
        assert_eq!(fire_step(&mut m, dt, 500, |t| 1.0 + 0.1 * t), None);
        // ...and an exponential approach fires only once it has flattened —
        // well after the two-window minimum a constant signal needs.
        let mut m = PlateauMonitor::new(1.0, CONVERGED_TOL);
        let fired = fire_step(&mut m, dt, 100_000, |t| 1.0 - (-t / 20.0).exp()).unwrap();
        assert!(fired > 10 * SAMPLES_PER_WINDOW as u64, "fired at step {fired}");
    }

    /// Non-finite samples cannot latch a verdict or poison the mean.
    #[test]
    fn non_finite_samples_drop_the_window() {
        let mut m = PlateauMonitor::new(1.0, CONVERGED_TOL);
        let dt = m.sample_dt();
        assert_eq!(fire_step(&mut m, dt, 1000, |_| f64::NAN), None);
        assert!(m.last_mean().is_none());
        // A NaN partway through does not stop a later honest plateau.
        let mut m = PlateauMonitor::new(1.0, CONVERGED_TOL);
        let fired = fire_step(&mut m, dt, 1000, |t| if t < 5.0 { f64::NAN } else { 1.0 });
        assert!(fired.is_some());
    }

    /// The verdict latches, and only `reset` clears it.
    #[test]
    fn verdict_latches_until_reset() {
        let mut m = PlateauMonitor::new(1.0, CONVERGED_TOL);
        let dt = m.sample_dt();
        let mut t = 0.0;
        while !m.converged() {
            t += dt;
            m.update(t, 1.0);
        }
        for _ in 0..1000 {
            t += dt;
            assert!(m.update(t, 1.0 + 9.0 * t)); // a runaway cannot un-latch it
        }
        m.reset(t);
        assert!(!m.converged());
        assert_eq!(m.windows(), 0);
    }

    /// A degenerate domain (zero or non-finite length) never fires rather than
    /// firing instantly on a zero-length window.
    #[test]
    fn degenerate_window_never_fires() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut m = PlateauMonitor::new(bad, CONVERGED_TOL);
            assert_eq!(fire_step(&mut m, 1.0, 10_000, |_| 1.0), None, "window {bad}");
        }
    }

    #[test]
    fn window_scales_with_the_domain() {
        let short = window_time_for(46.4, 1.24);
        let long = window_time_for(282.0, 1.24);
        assert!((long / short - 282.0 / 46.4).abs() < 1e-12);
        // 0.5 transit of the 46.4 r_t demo domain at gamma 1.24.
        assert!((short - 0.5 * 46.4 / reference_speed(1.24)).abs() < 1e-12);
    }

    #[test]
    fn relative_drift_is_scale_free_and_signed_safely() {
        assert_eq!(relative_drift(0.0, 0.0), 0.0);
        assert_eq!(relative_drift(0.0, 1.0), 1.0);
        assert_eq!(relative_drift(-1.0, 1.0), 2.0);
        assert!((relative_drift(1000.0, 1001.0) - 1.0 / 1001.0).abs() < 1e-15);
        assert!(relative_drift(1e-30, 0.0).is_finite());
    }
}
