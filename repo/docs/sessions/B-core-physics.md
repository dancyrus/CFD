# Session B — `cfd-core/src/physics.rs`

**Budget: 95 minutes.**

Every one of your deliverables is a pure function over slices with its own test. **You need nothing from session A to test any of them.** Write the test as you write the function and run it immediately; do not wait for merge.

## Read first

`docs/physics-reference.md` (§3, §4, §5 especially), `docs/contract.md`, `cfd-contract/src/lib.rs`, `cfd-contract/src/kernels.rs`.

## Rules

- You own **`cfd-core/src/physics.rs` and nothing else.**
- Never modify `cfd-contract/`, `cfd-core/src/lib.rs`, or `cfd-core/src/step.rs`. If the contract is wrong, print `CONTRACT CHANGE REQUEST:` with the exact diff and stop.
- All state is non-dimensional, chamber-referenced. `R = 1`, `p = rho*T`, chamber is exactly `(1, 0, 0, 1)`.
- Every reduction in `f64`.

## What you build

### 1. Wall flux — `wall_flux_z`, `wall_flux_r`

```
fn wall_flux_z(w: Prim, sgn: Real, gamma: Real) -> Cons {
    a  = sqrt(gamma * w.p / w.rho)
    un = sgn * w.u_z                     // outward-from-fluid normal velocity
    SL = -(abs(un) + a)                  // Davis speed of the mirrored Riemann problem
    ps = w.p + w.rho * un * (un - SL)    // HLLC star pressure at S_M = 0
    ps = max(ps, p_min)                  // a receding wall can drive this negative
    [0.0, sgn*ps, 0.0, 0.0]
}
```

`wall_flux_r` is identical, returning `[0, 0, sgn*ps, 0]` with `un = sgn * w.u_r`.

**Do not implement this by calling `hllc_flux` with a mirrored state.** The `S_M = 0` cancellation is exact in real arithmetic but depends on the compiler not reassociating `S_R = -S_L` under FMA contraction. Special-casing it is three lines and gives bit-exact zero mass and energy flux through the wall no matter what the optimizer does. Thrust is a momentum balance; a wall that leaks mass makes every reported number meaningless.

Sign check: for `un > 0` (flow into the wall) `ps = p + rho*un*(2*un + a) > p`. For `un < 0` (receding) `ps = p - rho*|un|*a`, the acoustic expansion. Correct both ways.

Test: assert components 0 and 3 are **exactly** `0.0` for a hundred random states.

### 2. `fill_ghosts`

Four boundary kinds.

**Axis (r = 0):** two mirror ghost rows, `(rho, u_z, -u_r, p)`. Needed only to feed the MUSCL slope at row 0; the axis face itself carries zero flux.

**Subsonic stagnation inlet** (chamber face, rows inside the chamber radius):

```
R_minus = u_i - 2*a_i/(g-1)
A       = (g+1)/(g-1)
D       = (g+1)*a0^2/(g-1) - (g-1)*R_minus^2/2
D       = max(D, 0.0)                    // MANDATORY. Goes negative under transient
                                         // reverse flow; without it you NaN before
                                         // the first frame renders.
a_b = (-R_minus + sqrt(D)) / A           // larger root
u_b = max(R_minus + 2*a_b/(g-1), 0.0)
v_b = 0.0                                // axial injection. Undefined tangential
                                         // velocity at an inlet is a classic
                                         // silent-garbage source.
p_b = p0 * (a_b^2/a0^2)^(g/(g-1))
rho_b = g*p_b / a_b^2
```

Non-dimensional: `a0^2 = gamma * t0`, and `t0 = p0 = 1`.

Test: feed an interior state, assert `p_b <= p0` and `0 <= u_b`, and assert no NaN for interior states with `u_i` swept negative.

**Downstream outflow.** Supersonic (`|u| >= a`): copy the interior cell. All four characteristics exit, so this is exact and non-reflecting. Subsonic: impose `p_a`, extrapolate entropy and R⁺:

```
rho_b = rho_i * (p_a/p_i)^(1/g);  a_b = sqrt(g*p_a/rho_b)
u_b   = (u_i + 2*a_i/(g-1)) - 2*a_b/(g-1);  v_b = v_i;  p_b = p_a
```

**No sponge on the downstream boundary.** The core is supersonic there and a sponge would corrupt it.

**Radial far field:** same construction with the radial normal; if flow is entering (`v_i < 0`), use the ambient reservoir.

### 3. `apply_sponge` — dt-based, not per-step

```
L = cells (24)
for each cell in the outer L rows:
    s = depth_into_sponge / L
    sigma_max = 12.0 * a_ambient / (L * dr)      // units of 1/time
    U -= dt * sigma_max * s*s * (U - U_ambient)
```

The obvious per-step form (`U = (1-0.05*s^2)*U + ...`) is 4–6× too weak and resolution-dependent: an outgoing wave crossing 16 cells accumulates only `Σσ ≈ 0.9`, giving 41% one-way transmission and about 17% of the wave returning after reflecting off the far boundary. That shows up as a standing pattern in the plume and gets misdiagnosed as a solver bug. The dt-based form above targets `∫σ dt ≈ 4` — 1.8% one-way, under 0.1% round-trip.

Test: a plane wave launched into the sponge loses at least 98% of its amplitude crossing it.

Also emit a diagnostic when the plume (`p > 1.5*p_a`) reaches the sponge entry.

### 4. `carbuncle_mask`

```
Omega(i,j) = min(p[i-2..=i+2, j]) / max(p[i-2..=i+2, j])
mask[i-1..=i+1, j] = true  where Omega < 0.7
```

The caller uses HLL for the **radial** fluxes of masked cells and HLLC everywhere else; axial fluxes always use HLLC, because contact resolution is what keeps the plume boundary sharp.

The window is ±2 cells, not ±1: a captured shock is 2–3 cells wide, so a ±1 window can read intermediate pressures on both sides and miss the shock's own cell. The mask spans three cells, not one, or the carbuncle nucleates at the shock foot.

At γ = 1.24 the 0.7 threshold fires above about M 1.17. Carbuncle only afflicts M ≳ 2 (Ω < 0.23), so 0.7 is conservative on purpose.

### 5. `mach_from_area_ratio` — NASA b4wind

```
P = 2/(g+1);  Q = 1 - P
subsonic:   p=P, q=Q, Rr=ar^2
supersonic: p=Q, q=P, Rr=ar^(2Q/P)
E = 1/q;  a = p^E;  r = (Rr - 1)/(2a)
X0 = 1 / ((1+r) + sqrt(r*(r+2)))
Newton: f(X) = (p+q*X)^E - Rr*X,  f'(X) = (p+q*X)^(E-1) - Rr
M = sqrt(X) subsonic;  M = 1/sqrt(X) supersonic
```

**Guard `|ar - 1| < 1e-6` and return `1.0` immediately.** At `ar = 1` exactly, `f'(X) = 0` at the root — it is a double root — so Newton returns NaN. The throat cell is set analytically, but `quasi1d_init` sweeps *every* column and any column whose open radius lands within roundoff of the sonic area hits this. This is a real bug, not a theoretical one.

Eight fixed Newton iterations is enough; the closed-form initial guess lands in (0, 1] on both branches without a bracketing fallback. Test the round-trip `M -> ar -> M` over γ ∈ {1.2, 1.24, 1.4}, M ∈ [0.005, 15], both branches, and **assert 1e-11, not 3e-13** — the tighter figure holds only at γ = 1.4; at γ = 1.2 near sonic it reaches 2e-12.

### 6. `quasi1d_init`

```
open radius per column:  r_w(i) = sqrt( 2*dr * sum_j (1 - frac[i][j]) * r_j )
```

Note the r-weighting and the square root. A linear sum is dimensionally wrong and gives the wrong quasi-1D Mach number, throwing away part of the speedup this function exists for.

Then: find `i_throat` at the minimum area; subsonic branch upstream, supersonic downstream, `M = 1` analytically at the throat; fill each column from the isentropic relations; ambient outside the nozzle and downstream of the exit; blend over 4 cells axially at the exit plane so step 1 does not launch a delta-function shock.

If the area has no interior minimum — an arbitrary drawn blob — fall back to ambient everywhere. Do not crash.

This is worth roughly 3× fewer steps to steady state than a cold start, and the app needs the quasi-1D solution anyway for the textbook-comparison overlay, so it is nearly free.

### 7. `apply_geometry_change` — the flip ledger

Both original design passes omitted this and it will silently corrupt the conservation test.

```
for each cell:
    vol = r_j * dr * dz                       // Grid::cell_vol
    fluid -> solid: ledger.mass   -= rho[c]*vol
                    ledger.energy -= E[c]*vol
                    valid[c] = false
    solid -> fluid: valid[c] = false; queue.push(c)

BFS refill, at most 8 passes:
    for c in queue with >= 1 valid fluid neighbour:
        rho[c] = mean(rho over those neighbours)
        p[c]   = mean(p over those neighbours)
        u[c] = 0;  v[c] = 0                   // START AT REST
        ledger.mass += rho[c]*vol; ledger.energy += E[c]*vol

remaining invalid cells -> ambient
```

`u = v = 0` in newly opened cells is not laziness. A cell opened next to a Mach-3 plume that inherits its neighbour's velocity fires a shock into the new cavity.

The ledger is what separates "conservation drift" from "the user drew a hole." Test T2 asserts `|Δmass_total − ledger.mass| < tol`, not `|Δmass_total| < tol`. Without it, every geometry edit is a false test failure and someone spends an hour chasing a non-bug.

Rate-limit: this is drained at the top of `step()`, never inside an RK stage, and at most once per 100 ms or on pointer-up.

## Done when

`cargo test -p cfd-core physics::` is green, with at least one test per function above.

## Known traps

- Forgetting `D = max(D, 0)` in the inlet. NaN before the first frame.
- Calling `hllc_flux` for the wall instead of special-casing it. Works until it does not.
- The per-step sponge form. Looks reasonable, is 4–6× too weak.
- Linear open-radius sum in the init. Looks right, is dimensionally wrong.
- Newton at `ar = 1`. NaN.
