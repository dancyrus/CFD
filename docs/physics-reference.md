# CFD Sandbox — Physics and Numerics Reference

The authority on every formula, constant and tolerance in this build. Every session reads this before touching numerics.

Items marked ✓ were independently recomputed by a separate verification pass — exact Riemann solver, Taylor–Maccoll ODE integration, symbolic identity checks, 18,000-sample Newton sweeps. Items marked ⚠ are corrections to an earlier draft; the wrong version is described so nobody reintroduces it.
---

## 0. Locked decisions

| Decision | Value | Why |
|---|---|---|
| Grid | Graded tensor-product axisymmetric (z, r): arbitrary per-axis cell-edge lists, anisotropic; uniform is the special case | Body-fitted meshing is build step 10 in the long-term architecture; it is not a PoC item. Grading (docs/work-orders/grid-grading.md) buys a readable plume without the ~14× cost of enlarging the uniform grid; the spacing rule lives in cfd-geom, never in cfd-core |
| Geometry | Solid volume fraction, exact sub-cell area | Both the parametric nozzle and drawn strokes rasterize into one field, so one solver serves both |
| Wall | Sharp threshold at fraction ≥ 0.5, reflecting wall flux | Bit-exact zero mass flux through the wall, zero timestep penalty, works on a one-cell-thick stroke |
| Precision | `f32` in the hot loop, `f64` in every reduction | See §7 |
| Units | Non-dimensional internally, SI at the `Snapshot` boundary | See §7 |
| γ | 1.24 in the demo case, 1.4 hard-coded in the verification tests | See §6 |
| Scheme | MUSCL(minmod) + HLLC + SSP-RK2, CFL 0.4 | |
| Aperture weighting | OUT | See §3 |
| Local timestepping | OUT | Deferred; the turbo control in §9 buys most of it for five lines |

---

## 1. Governing equations

Conserved `U = (ρ, ρu, ρv, E)`, z axial, r radial, u axial velocity, v radial velocity, `E = p/(γ−1) + ½ρ(u²+v²)`.

```
∂U/∂t + ∂F/∂z + (1/r)∂(rG)/∂r = (0, 0, p/r, 0)

F = (ρu, ρu²+p, ρuv, u(E+p))
G = (ρv, ρuv, ρv²+p, v(E+p))
```

The grid is a tensor product of per-axis cell-edge lists (graded or uniform). Faces at the edges `r_{j±1/2}`; per-cell widths `dz_i`, `dr_j`. Two distinct cell radii exist and they differ by up to `dr/6` (~4% on graded rows):

- `r_j = (r_{j−1/2} + r_{j+1/2})/2`, the **arithmetic mean of the face radii** — the exact volume radius (`r_j·dr_j` is the true shell area). Volumes, areas and the update below use this.
- the **shell volume centroid** `(2/3)(r_hi³−r_lo³)/(r_hi²−r_lo²)` — where a cell average of a linear-in-r field actually sits. Radial RECONSTRUCTION uses this (in z, cell centres/midpoints).

Update:

```
dU_i,j/dt = −(F_{i+1/2} − F_{i−1/2})/dz_i
            − [ r_{j+1/2}·(G_{j+1/2} − S_j) − r_{j−1/2}·(G_{j−1/2} − S_j) ] / (dr_j·r_j)

S_j = (0, 0, (p̂_{j−1/2} + p̂_{j+1/2})/2, 0)
```

where `p̂_{j±1/2}` are the **interface pressures the Riemann solver sampled at the two radial faces** (`hllc_flux_p`/`hll_flux_p`; a wall face contributes its star pressure; identical face states contribute their own p). **Write it exactly as bracketed above.**

Why the face-pressure source (work-order item c): since `(1/r)∂(rp)/∂r − p/r ≡ ∂p/∂r`, the pressure part of the radial momentum equation is a plain gradient — it should not be discretized through the geometric r-weighting at all. With `S_j` built from face pressures and `r_j` the arithmetic face mean, the bracket is algebraically identical to *conservative r-weighted advective flux difference plus `(p̂_hi − p̂_lo)/dr_j`*: no cell-centre radius enters the pressure balance, and a linear-in-r pressure field is differenced exactly on ANY radial grid (ref arXiv 1701.04834). The earlier uniform-grid form with `S_j = (0,0,p_j,0)` (cell-centre pressure) is the special case that was only exact for uniform p; substituting the cell-centre pressure at the axis row mis-differences a linear `dp/dr` by a factor 3.

The axis face (`j = 0` lower face, `r_{−1/2} = 0`): the flux is weighted by zero, but its **interface pressure still anchors the source** — it comes from the mirrored Riemann problem against the axis ghost, whose star pressure is the symmetry-plane pressure.

In `f32` a separated flux-difference-plus-source form leaves a residue of order `1e-7 × 40p × dt` per step at j = 0, which accumulates into a faint axis artifact over a cold start. Inside one bracket the identical operands cancel bit-exactly.

Three properties this form gives you:

- The axis face at j = 0 has `r_{−1/2} = 0`, so it carries zero flux by construction. The axis is a wall because of the geometry, not because of a boundary condition.
- Exactly well-balanced **for the quiescent uniform state at any radial grading**: for uniform p every face returns `p̂ = p` and `G − S` vanishes identically, at every j including 0. Verified bit-exact at growth ratio 1.2 (`well_balanced_graded`). Do not claim well-balancing for anything stronger.
- `r_j ≥ dr_0/2 > 0` always, so nothing divides by zero.

Axis boundary: two mirror ghost rows, `(ρ, u, −v, p)`, at mirrored radii (ghost widths mirror the interior). Needed to feed the MUSCL slope at j = 0 and the axis-face interface pressure.

**Non-uniform reconstruction** (work-order items a–b): slopes are computed from one-sided difference QUOTIENTS `Δ± = ΔW/Δx` over the actual positions, and face states extrapolate over the actual centre-to-face distances. minmod needs no change beyond that. The unlimited slope must be the full quadratic-fit gradient `(h_l·Δ+ + h_r·Δ−)/(h_l + h_r)` — including the cell's own value term; the naive average `(Δ− + Δ+)/2` or plain distance weighting `(W_{i+1} − W_{i−1})/(h_l + h_r)` is formally first order off-uniform and injects an O(ratio−1) fraction of upwind dissipation. Van Leer generalizes to the weighted harmonic `Δ−Δ+(h_l+h_r)/(h_l·Δ− + h_r·Δ+)`. All reduce exactly to the uniform formulas at equal spacing. Guards: `t1_freestream_graded`, `well_balanced_graded`, `t4_sod_graded` (acceptance.rs) run at growth ratios 1.15–1.2.

---

## 2. Carbuncle control

The Mach disk is a strong grid-aligned shock sitting on the axis, which is the textbook HLLC shock-instability trigger.

```
Ω(i,j) = min(p[i−2..i+2, j]) / max(p[i−2..i+2, j])
if Ω < 0.7:  use HLL for the RADIAL fluxes of cells (i−1, i, i+1) at this row
else:        use HLLC
Axial fluxes always use HLLC — contact resolution is what keeps the plume boundary sharp.
```

⚠ Two corrections to the first draft: the sensor window is ±2 cells, not ±1 (a captured shock is 2–3 cells wide, so a ±1 window reads intermediate pressures on both sides and misses the shock's own cell), and the switch applies to three cells, not one (or the carbuncle nucleates at the shock foot).

At γ = 1.24 the 0.7 threshold fires for shocks above about M 1.17. Carbuncle only afflicts M ≳ 2 (Ω < 0.23), so 0.7 is conservative. Keep it.

---

## 3. Immersed boundary — the exact update

Rejected: diffuse BDIM body forcing. Not for the usual stiffness reason (BDIM is a convolution reconstruction, not Brinkman penalization, and its authors state there is no timestep penalty). It is rejected because the proposed recipe — project out normal velocity, keep p, rebuild ρE — changes momentum and energy outside the flux divergence, and **thrust is a momentum balance**. You cannot compute a defensible thrust from a non-conservative wall. It also needs walls ≥ 6 cells thick to stop mass leaking, which a user's 2-pixel stroke cannot satisfy.

Also rejected: aperture weighting. The proposal gave only the axial closure; the radial one in the r-weighted form must satisfy `Σ_f r_f A_f n_f + r_w A_w n_w = 0` discretely or free-stream preservation fails in a way that looks exactly like a wall leak. Not a 30-minute job.

```
// once, on geometry change
frac[c] : f32          // EXACT sub-cell solid area fraction — see §8 T0
solid[c] : bool = frac[c] >= 0.5

// per RK stage
// 1. primitives, fluid cells only. Solid cells are never read.

// 2. slopes, per primitive variable, minmod
sz[c] = (solid[c−1] || solid[c+1]) ? 0 : minmod(W[c]−W[c−1], W[c+1]−W[c])
sr[c] = (solid[c−nz] || solid[c+nz]) ? 0 : minmod(...)
// A fluid cell touching a wall is piecewise-constant in the wall-normal direction.
// That is the entire stencil degradation. Nothing else is needed, and it is what
// makes step 3 unambiguous. Price: first order in a one-cell band at the wall.

// 3. z-faces between c=(i,j) and d=(i+1,j)
(solid[c], solid[d]):
  (true , true ) -> F = 0
  (false, false) -> qL = W[c] + 0.5*sz[c]; qR = W[d] − 0.5*sz[d]
                    (qL,qR) = positivity_fallback(qL, qR, W[c], W[d])
                    F = hllc(qL, qR)
  (false, true ) -> F = wall_flux_z(W[c], +1)
  (true , false) -> F = wall_flux_z(W[d], −1)

fn wall_flux_z(W, sgn):
    a  = sqrt(gamma * W.p / W.rho)
    un = sgn * W.u                    // outward-from-fluid normal velocity
    SL = −(abs(un) + a)               // Davis speed of the mirrored Riemann problem
    ps = W.p + W.rho * un * (un − SL) // HLLC star pressure at S_M = 0
    ps = max(ps, P_MIN)               // a receding wall can drive this negative
    return [0.0, sgn*ps, 0.0, 0.0]    // mass, r-momentum, energy all EXACTLY zero

// 4. r-faces identical, returning [0, 0, sgn*ps, 0], un = sgn*W.v, flux weighted by r_{j±1/2}
```

Do **not** implement this by calling `hllc()` with a mirrored state. The `S_M = 0` cancellation is exact in real arithmetic but depends on the compiler not reassociating `S_R = −S_L` under FMA contraction. Special-casing the wall face is three lines and gives bit-exact zero mass flux without trusting the optimizer.

Sanity on the sign: for `u_n > 0` (flow into the wall) `ps = p + ρu_n(2u_n + a) > p`. For `u_n < 0` (receding) `ps = p − ρ|u_n|a`, the acoustic expansion. Correct both ways.

### Cells that flip when the user edits geometry mid-run

Both original passes omitted this and it will silently corrupt the conservation test.

```
// drained at the TOP of step(), never inside an RK stage.
// rate-limited to one apply per 100 ms or on pointer-up (the fill is a multi-pass BFS).
on GeometryChanged(new_frac):
  for c:
    vol = 2π·r_j·dr·dz
    fluid -> solid: ledger.mass −= ρ[c]·vol; ledger.energy −= E[c]·vol; valid[c] = false
    solid -> fluid: valid[c] = false; queue.push(c)
  // BFS fill, at most 8 passes
  for c in queue with >=1 valid fluid neighbour:
    ρ[c] = mean(ρ over those neighbours); p[c] = mean(p over them)
    u[c] = 0; v[c] = 0                   // START AT REST
    ledger.mass += ρ[c]·vol; ledger.energy += E[c]·vol
  remaining invalid cells -> ambient
  recompute dt; reset the convergence monitor
```

`u = v = 0` in newly opened cells is not laziness. A cell opened next to a Mach-3 plume that inherits its neighbour's velocity fires a shock into the new cavity.

The ledger is what separates "conservation drift" from "the user drew a hole." Test T2 asserts `|Δmass_total − ledger.mass| < tol`, not `|Δmass_total| < tol`. Without it every geometry edit is a false test failure.

---

## 4. Boundary conditions

**Subsonic stagnation inlet** (verified term-for-term against NASA/TM-2011-217181). Given `p0`, `T0`, `a0² = γT0` non-dimensional, and interior neighbour state:

```
R⁻  = u_i − 2a_i/(γ−1)
A   = (γ+1)/(γ−1)
D   = (γ+1)a0²/(γ−1) − (γ−1)R⁻²/2
D   = max(D, 0)                       // ⚠ goes negative under transient reverse flow;
                                      //    without this clamp you NaN before the first frame
a_b = (−R⁻ + sqrt(D)) / A             // larger root
u_b = max(R⁻ + 2a_b/(γ−1), 0)
v_b = 0                               // ⚠ the original spec never said; undefined tangential
                                      //    velocity at an inlet is a classic garbage source
p_b = p0·(a_b²/a0²)^(γ/(γ−1))
ρ_b = γ·p_b / a_b²
```

**Downstream outflow.** Supersonic (`|u| ≥ a`): copy the interior cell. All four characteristics exit, so this is exact and non-reflecting. Subsonic: impose `p_a`, extrapolate entropy and R⁺.

```
ρ_b = ρ_i·(p_a/p_i)^(1/γ);  a_b = sqrt(γp_a/ρ_b)
u_b = (u_i + 2a_i/(γ−1)) − 2a_b/(γ−1);  v_b = v_i;  p_b = p_a
```

**No sponge on the downstream boundary.** The core is supersonic there and a sponge would corrupt it.

**Radial far-field.** Same construction with the radial normal; if flow is entering (`v_i < 0`) use the ambient reservoir. Plus a sponge.

⚠ **The sponge coefficients in the first draft were 4–6× too weak and resolution-dependent.** `σ = 0.05·s²` applied per *step* over 16 cells gives `Σσ ≈ 0.90` for a wave crossing at the default grid — 41% one-way transmission and about 17% of the wave returning after reflecting off the far boundary. That is a visible standing pattern in the plume and it will be misdiagnosed as a solver bug. Being per-step also makes it 1.8× stronger on the measure grid. Use a dt-based form targeting `∫σ dt ≈ 4`:

```
U −= dt · σ_max · s² · (U − U_ambient),   s = (depth into sponge)/L_sponge
L_sponge = PHYSICAL depth of the outer 24 rows (on a graded far field those
           rows are wide; an index-based depth misplaces the profile)
σ_max    = 12·a_ambient / L_sponge        // units of 1/time
```

That gives 1.8% one-way transmission and under 0.1% round-trip return. Add a diagnostic: warn if the plume (`p > 1.5 p_a`) reaches the sponge entry.

---

## 5. Initialization, floors, timestep

**Quasi-1D isentropic init** in the nozzle interior, ambient elsewhere, blended over 4 cells at the exit plane. Roughly 3× fewer steps to steady than a cold start, and the app needs the quasi-1D solution anyway for the textbook-comparison overlay, so it is nearly free.

⚠ The open-radius formula in the first draft was dimensionally wrong (`dz·Σ_j`) and unweighted. Correct (per-cell `dr_j` inside the sum on a graded grid):

```
r_w(i) = sqrt( 2·Σ_j (1 − frac[i][j])·r_j·dr_j )
```

Linear summing gives the wrong quasi-1D Mach number by the r-weighting and throws away part of the speedup.

**Area–Mach inversion — NASA b4wind.** Verified: worst round-trip error 4.75e-13 over γ ∈ {1.2, 1.24, 1.4}, M ∈ [0.005, 15], both branches, with 8 fixed Newton iterations. The closed-form initial guess landed in (0, 1] on all 18,000 samples; no bracketing fallback needed.

```
P = 2/(γ+1);  Q = 1 − P
subsonic:   p=P, q=Q, Rr=(A/At)^2
supersonic: p=Q, q=P, Rr=(A/At)^(2Q/P)
E = 1/q;  a = p^E;  r = (Rr − 1)/(2a)
X0 = 1 / ((1+r) + sqrt(r(r+2)))
Newton on  f(X) = (p+qX)^E − Rr·X,  f'(X) = (p+qX)^(E−1) − Rr
M = sqrt(X) subsonic;  M = 1/sqrt(X) supersonic
```

⚠ **`f'(X) = 0` exactly at the root when `A/At = 1`, so Newton returns NaN.** The throat cell is set analytically, but the init sweeps *every* column and any column whose open radius lands within roundoff of the sonic area hits this. Guard on `|A/At − 1| < 1e-6 → M = 1`, not on exact equality.

⚠ Assert the round-trip to **1e-11**, not 3e-13. The tighter figure holds only at γ = 1.4; at γ = 1.2 near sonic it reaches 2e-12.

**Positivity.** ⚠ Two fixes to the first draft. (a) A face-level fallback must replace **both** reconstructed states with their cell averages — a one-sided fallback makes the flux multivalued and breaks conservation. (b) Face-level fallback does not stop the *cell average* going negative after the update, so add a cell-level pass:

```
face level: if either reconstructed state has ρ ≤ ρ_min or p ≤ p_min,
            use cell averages on BOTH sides of that face
cell level: after each RK stage, if ρ[c] ≤ ρ_min or p[c] ≤ p_min,
            redo cell c with all four faces first-order;
            if still bad, clamp and increment floor_counter
```

First-order HLLC/HLL with Davis speeds is provably positivity-preserving at CFL ≤ 0.5 and you run at 0.4, so the cell-level redo is sound. Without it the vacuum end of the altitude slider trips the floor counter, and the product rule says nothing displays when that counter is nonzero — you would ship an app that refuses to show numbers.

**Floors** (non-dimensional, chamber-referenced): `ρ_min = 1e-8`, `p_min = max(1e-6·p0, 1e-4·p_a)`. ⚠ The absolute floor has been raised twice: 1e-9 → 1e-8 because at the 50 km ceiling `1e-4·p_a` sat at the old clamp and the plume core rode the floor; 1e-8 → 1e-6 with the real-engine presets, so the floor tracks chamber pressure across the 8.6–300 bar preset range. Altitude ceiling 58 km — that is where the thinnest ambient still clears the floor for the highest-pressure preset (Raptor 2, 300 bar: p_a(58 km) ≈ 27 Pa ≈ 1e-6·p₀). Above the cap the UI switches to a **labelled vacuum mode** with back pressure fixed at 3e-5·p₀ — 30× the floor, an empirical margin from the Merlin Vac (ε = 165, exit pressure ≈ 8e-5·p₀) probes: at 2× the floor its plume rides the floor at steady state, at 10× the startup contact stretches to step ~1900, at 30× floor contact is confined to the cold-start front (last activation step 796, zero after — see §13's quarantine rule). Any atmospheric ambient below 3e-5·p₀ is clamped to it for the same reason.

**Timestep.** `w_max = max[(|u|+a)/dz_i + (|v|+a)/dr_j]` over fluid cells with the LOCAL cell widths, `dt = 0.4/w_max`, recomputed every step. Fold the reduction into the same rayon pass that computes primitives — it is one extra streaming pass and effectively free. A nozzle startup swings the max wave speed by 5×, so a fixed dt either diverges during startup or wastes 5× at steady state.

---

## 6. Gas model

**Demo case: γ = 1.24, R = 378 J/(kg·K) (Mw ≈ 22), p₀ = 5 MPa, T₀ = 3200 K, ε = 8, r_t = 50 mm.**

γ = 1.4 is not acceptable for the demo. Verified sensitivity at ε = 8, holding everything else:

| γ | M_exit | p_e | C_f (SL) | Isp_SL |
|---|---|---|---|---|
| 1.20 | 3.122 | 84.3 kPa | 1.551 | 268.2 s |
| **1.24** | **3.224** | **76.2 kPa** | **1.531** | **261.7 s** |
| 1.40 | 3.677 | 51.1 kPa | 1.468 | 240.3 s |

Going from 1.24 to 1.4 moves exit Mach 14% and Isp 10.4%, and moves the ideal-expansion pressure ratio from 65.6 to 97.8 — a 49% shift in where the altitude slider's design point sits. The slider's whole purpose is finding that point. A label in the UI does not fix a mislabeled axis.

The single-gas simplification (exhaust γ applied to the ambient air too) is fine here: the ambient is quiescent, the jet boundary is a pressure match, and pressure in still gas is γ-independent. The 6% ambient sound-speed error touches only the ambient CFL contribution. A species-transport mixture γ would change the flux fragment, the conserved-to-primitive fragment, the HLLC wave speeds and the field count, and brings the Abgrall interface-pressure-oscillation problem. Out.

**Why p₀ = 5 MPa specifically.** At γ = 1.24, ε = 8, the perfect-expansion ambient is 76.2 kPa — about 2.3 km altitude. So the slider sweeps mildly over-expanded at sea level (p_e/p_a = 0.75), through the design point just above it, into under-expanded and free expansion above. Since the separation threshold is 0.40, **the entire slider range is inside the regime an inviscid solver can be trusted for.** That is worth preserving; it is why the demo is honest.

**Verification tests use γ = 1.4 as a hard-coded literal**, because the exact references for Sod, θ-β-M and Taylor–Maccoll only exist there. Test T8 (nozzle vs isentropic) recomputes its reference from `case.gas.gamma`, so it works at any γ.

Note the digitized Rao θ_n/θ_e table was produced at a specific γ (around 1.23–1.25). Label it in code with that γ or the divergence factor is quietly inconsistent.

---

## 7. Precision and units

**`f32` in the hot loop.** Verified precision analysis: digits remaining when recovering `p = (γ−1)(E − ½ρ|u|²)` is `7.2 − log₁₀(1 + γ(γ−1)M²/2)`. At M = 12 (worst realistic plume) you keep 5.9 significant digits; even at M = 100 you keep 4.05. Precision is not the risk.

⚠ Correct the reasoning, not the conclusion. The claim that f64 "halves" a bandwidth-bound kernel does not hold: at 22 Mcups × 2 stages × ~40 B/cell the kernel moves about 1.8 GB/s against ~68 GB/s of M1 bandwidth. It is latency and ALU bound on the sqrt, divides and branches in HLLC, not memory bound. The real f64 penalty is 1.2–1.5× from NEON lane count, and only where the code vectorizes, which branchy HLLC largely will not. f32 is still right — the hours of rewriting are worth more than 1.35× — but say why honestly.

**Non-dimensional internally, chamber-referenced:** `L_ref = r_t`, `p_ref = p₀`, `ρ_ref = ρ₀ = p₀/(R T₀)`, `u_ref = sqrt(p₀/ρ₀)`, `T_ref = T₀`. So R = 1, `p = ρT`, and the chamber state is exactly (1, 0, 0, 1).

⚠ Correct the reasoning here too. "Keeps values ≤ 1 for f32" is a misconception — IEEE-754 relative precision is scale-invariant across 76 decades. Non-dimensionalizing buys zero precision. Do it for three real reasons: the floors, the sponge coefficient and the positivity thresholds are only portable if the state is O(1) (in SI they need re-deriving every time p₀ changes); the chamber state being exactly (1,0,0,1) makes T1 testable against exact literals; and the T4 Sod threshold is a non-dimensional literature number that must be compared in the same units.

**Every reduction in `f64`.** Mandatory, not advisory. An f32 sum over the 64k default grid has RMS noise 1.5e-5 and a sequential worst case of 3.8e-3 — larger than the T4 Sod pass threshold itself. On the 256k measure grid the worst case is 1.5e-2. Mass, momentum, energy, thrust integrals, L1 norms: all f64.

Put `pub type Real = f32;` in the contract crate so it is a one-line flip if T3 or T4 disappoint.

---

## 8. Grid sizing

The three requirements — a domain long enough for a Mach disk, a throat resolved enough for defensible mass flow, and interactive frame rates on an M1 Air — cannot all be met. Cost scales as `L_z²·L_r/(dz·dr²)`, so halving dr is 4× the work, not 2×.

The lever nobody used: **dz and dr need not be equal.** dt is set by `(|u|+a)/dz + (|v|+a)/dr`; with dr ≪ dz the radial term dominates, so widening dz is nearly free in dt while linearly cutting cells.

⚠ Superseded by the grid-grading work order: an earlier draft rejected geometric stretching because it "breaks exact area rasterization, the `r_j = (j+0.5)dr` identity that makes §1 well-balanced, and uniform-grid slope limiting." All three objections were about *implementations that assume uniformity*, not about stretching itself, and all three were closed properly: the rasterizer clips against the actual cell edges (T0 unchanged), §1's face-pressure source form is well-balanced at any radial grading (bit-exact, `well_balanced_graded`), and the reconstruction uses the full unequal-spacing stencils (§1). The spacing rule — hold base resolution across the solid span plus a margin, grade geometrically at 1.05 beyond, cap cell growth at 6× base — lives in `cfd-geom::grade_from_solid`; the solver just takes edge lists. Base resolution held across the geometry and the plume core (radially to 3.54 × the wall's outer radius, this section's compact sizing) keeps every §8 number below intact on the held region, and dt is set by the held base spacing, so enlarging the domain does not shrink it.

⚠ **Reversal (configurable-domain work order, 2026-08):** the original decision below — trade plume length to protect throat resolution and interactivity — is deliberately reversed. At this stage of the project simulation quality outranks interactivity: a fine mesh in a large domain that takes minutes to converge is worth more than a coarse mesh that returns instantly and may not be accurate. The fixed domain tiers (Compact / Standard / Long, largest 125.6 × 16 r_t for the demo case) are gone; domain length, domain radius, base resolution (8–160 cells/r_t) and the dz/dr aspect are direct sidebar inputs, and the default domain is the largest shortcut, **282 × 32 r_t** — roughly 2× the old largest in both directions. The guard rails moved from the inputs to a live cost readout: total cells, estimated steps and wall clock (from a throughput calibration measured on the running machine, never a constant from another machine), estimated memory, an amber warning above ~30 minutes, and a blocking confirmation for settings that would exhaust memory.

| shortcut | extents (r_t) | demo-case graded grid | N_throat | dz / dr (r_t) |
|---|---|---|---|---|
| Preview | 46.4 × 10 | 142 × 200 = 28.4k cells | 20 | 0.145 / 0.05 |
| Standard | 141 × 16 | 251 × 246 = 61.7k cells | 20 | 0.145 / 0.05 |
| **Large (default)** | **282 × 32** | 413 × 300 = 123.9k cells | 20 | 0.145 / 0.05 |

(Historic reference, uniform grids: Interactive 320 × 200 = 64k over 46.4 × 10; Measure 640 × 400 = 256k, badged ±2.5%. Steps/s ranges from the original plan — 150–350 M1 / 400–860 PC at 64k — bracketed an optimistic 5.0 Mcups/core kernel against a 0.35–0.6× derate; measured figures now live in `docs/results/`.)

Domain anatomy at Preview: 11 r_t of nozzle (2 chamber + converging cone + arcs = 4.13, diverging L_n = 6.87) plus 12.5 exit radii of plume. Radial hold = 3.54 r_e against a peak plume radius near 2.2 r_e, so 1.6× margin; the sponge is the outer 24 rows (physical depth — entry at 8.8 r_t on Preview, near r = 25 ≈ 4× the peak plume radius on the Large domain, clear of the plume edge either way).

⚠ The paragraph below described the UNIFORM-grid tradeoff and set the historic Compact domain; the graded grid reopened it and the reversal above finished the job. The far-field diamonds get coarser axial cells (capped at 6× base), which is the honest price, and radial resolution through the plume core stays at base. The trust caveat still applies word for word: the downstream diamonds remain the least trustworthy pixels on the screen.

**What the uniform grid sacrificed: plume length, by 3.2×.** You get the Mach disk (around 6–9 r_e) plus the leading half of the first reflected cell. You do not get 4.5 visible diamonds. On a uniform grid that was the right thing to give up — the solver is inviscid, so there is no entrainment, the plume edge stays razor-sharp, and shock cells persist further downstream than in any photograph. **The downstream diamonds are the least trustworthy pixels on the screen.** The 40-exit-radii domain at the same throat resolution and UNIFORM spacing costs 204k cells and roughly two minutes per altitude drag on the M1 — that uniform enlargement stays dead; the graded Large domain is how the long plume is actually paid for.

⚠ Two premise corrections. Ashkenas & Sherman's `z_M/D = 0.67√(p₀/p_a)` is for a **sonic orifice** with D the orifice diameter; applied at a nozzle exit it gives 5.84 r_e, not 8.5. The true Mach-disk station is likely 6–9 r_e; the domain above covers the band. And Prandtl–Pack shock-cell spacing is a weakly-imperfectly-expanded linearization — at high underexpansion it is decorative. **Do not put a numeric shock-cell-spacing readout in the UI.**

**Display N_throat in cells next to the thrust readout, and go amber below 20.** The resolution input ranges 8–160 cells/r_t (superseding the earlier "clamp the sliders so N_throat ≥ 20": a coarse Preview-grade mesh is allowed, honestly badged — the mass-flow band, ±(100/N_throat)%, and the first-order staircase-wall biases, ≈13% thrust and ≈19% exit Mach at N_throat = 20 scaling as 20/N_throat, are computed from the live resolution, never quoted). Also worth knowing: at the default anisotropy the throat downstream arc (0.382 r_t) spans only 2.6 axial cells. That, not radial resolution, is the weakest link in T8. If T8 misses, the first lever is the aspect input, dz → 0.10 r_t (3.8 cells across the arc), not more radial cells.

⚠ Quantize `r_t` (or `dr`) so the throat lands exactly on a cell face. The ±1/N_throat mass-flow error is one geometric quantization of `r_t` against `dr`; aligning them collapses the systematic term to the second-order arc-curvature effect. About 10 lines, parametric nozzle only. Do not apply it to drawn geometry, which has no theoretical thrust to match.

---

## 9. Time to steady, and what the UI does about it

Two different questions were being answered by two different estimates, and both were right:

| Criterion | Physical time | Steps at the default grid |
|---|---|---|
| Supersonic core, barrel shock and Mach disk settled (≈5 plume transits) | ≈1.5 ms | ≈6,100 |
| Far field acoustically equilibrated (≈2 radial transits with a sponge) | ≈3.8 ms | ≈15,800 |

1.5 ms for a nozzle start transient is physically right, which is a useful check on the whole chain.

| | PC | M1 Air |
|---|---|---|
| Visual steady | 20–40 s | 50–100 s |
| Full steady | 55–105 s | 2.2–4.4 min |
| **Altitude change** (plume only) | **5–15 s** | **12–40 s** |

The altitude number is the one that matters, and it is small for a physical reason: **the nozzle interior is supersonic downstream of the throat, so ambient pressure cannot propagate upstream into it.** Changing ambient pressure re-equilibrates only the plume, 1–2 plume transits. This is why `set_environment` must never reset the field — that requirement is now load-bearing, not a nicety.

Exception: at the high-ambient end an over-expanded nozzle can push a shock inside the divergent section, and that shock moves slowly. The separation warning already flags that regime.

**Decision: the slider re-converges, asynchronously and visibly. It never resets the field, never blocks, and never interpolates between pre-converged states.** Three supports:

- `set_environment(p_a)` mutates only the outflow BC and the sponge target.
- The solver thread streams frames at 60 Hz regardless. The user *watches* the plume adjust, and that transient is the most compelling thing the app has.
- A **residual meter** (normalized L2 of ∂ρ/∂t) with a green "settled" dot below 1e-3, and **thrust, c\* and C_f greyed out whenever unsettled.** This is what makes the transient honest instead of a bug.
- A **turbo control** (1× / 4× / 16×) running N solver steps per rendered frame, gated by a frame-time budget. Five lines, and it buys most of what local timestepping would, without the 30 minutes and without destroying the transient's physical meaning.

Interpolating between pre-converged states is the worst option: more code, a lie, and it kills the only thing that looks alive.

---

## 10. Geometry

**Conical.** Upstream arc 1.5 r_t, downstream arc 0.382 r_t, converging half-angle 30°, contraction ratio 4 (r_c = 2 r_t).

```
throat arc (parameterize by local wall angle φ ∈ [0, α]):
    z(φ) = R_cd·sin φ;  r(φ) = r_t + R_cd·(1 − cos φ)
tangency:  zA = R_cd·sin α;  rA = r_t + R_cd·(1 − cos α)
cone:      r(z) = rA + (z − zA)·tan α  for zA ≤ z ≤ L_n
L_n = ( r_t(√ε − 1) + R_cd(sec α − 1) ) / tan α          ✓ identity verified in sympy
```

**Parabolic bell** (Rao approximation). Same 1.5 r_t converging arc; downstream arc 0.382 r_t running to wall angle θ_n; quadratic Bézier from there to the exit.

```
L_c15 = ( r_t(√ε − 1) + 0.382·r_t(sec 15° − 1) ) / tan 15°
L_n   = bell_percent · L_c15
N = ( 0.382 r_t sin θ_n ,  r_t + 0.382 r_t (1 − cos θ_n) )
E = ( L_n , r_t√ε )
m1 = tan θ_n; m2 = tan θ_e; C1 = N_r − m1·N_z; C2 = E_r − m2·E_z
Q = ( (C2−C1)/(m1−m2) , (m1·C2 − m2·C1)/(m1−m2) )
P(t) = (1−t)²N + 2t(1−t)Q + t²E
```

Verified for r_t = 50 mm, ε = 25, 80%: wall angle is exactly θ_n at t = 0 and exactly θ_e at t = 1 (error 0.00e+00 both), N_z < Q_z < E_z, monotone in both z and r. Guards the code needs: assert `θ_n > θ_e` (else division by zero) and assert `N_z < Q_z < E_z` (if bell_percent is too small for the area ratio, Q falls outside and the contour turns back on itself). Note `t` is not proportional to z.

**θ_n and θ_e** come from a digitized Rao table interpolated **log-linearly in ε** and linearly in bell percent. There is no published polynomial fit; every implementation uses a table. Verified: log-linear interpolation between ε = 20 (28.8°/9.0°) and ε = 30 (30.0°/8.5°) reproduces 29.4604°/8.7248° at ε = 25 exactly, confirming log-linear is the right axis (linear gives 29.400°/8.750°).

⚠ **Two circulating definitions of L_c15 differ by 0.337%.** The Huzel–Huang form above includes the throat-arc term; Aspirespace and the widely-copied `bell_nozzle.py` drop it. Pick the form above and state it, or "80% bell" is ambiguous.

⚠ The commonly-published digitized 60% column is non-monotonic (θ_n goes 37.1° at ε = 40 → 35.0° at ε = 50 — a digitization typo). It does not affect an 80% bell, but if the UI exposes bell percent, interpolating that row produces a nozzle whose divergence angle *decreases* with area ratio.

**Divergence factor:** `λ = (1 + cos θ_e)/2` for a bell, `(1 + cos α)/2` for a cone. Verified: the momentum-weighted integral `2∫₀¹ s·cos(θ_e s) ds` differs from `(1+cos θ_e)/2` by `−θ⁴/144`, which is 3.7e-6 at θ_e = 8.72°. The alternative `(1+cos((θ_n+θ_e)/2))/2` has no traceable primary source and gives a 2.75% loss against an accepted 0.5–1.5%. Note the two candidates differ by 1.1% and the product refuses to display efficiencies to better than 1%, so this is settled — do not let a session spend time on it.

---

## 11. Reports — one definition each

```
ṁ   = Σ_exit  ρ u_z · 2π r_j dr                      [exit-plane control surface]
F   = Σ_exit (ρ u_z² + p − p_a) · 2π r_j dr
C_f = F / (p₀ A_t)
c*  = p₀ A_t / ṁ
C_d = c*_ideal / c*,   c*_ideal = sqrt(γRT₀) / (γ·(2/(γ+1))^((γ+1)/(2(γ−1))))
```

Exit-plane control volume, not the wall-pressure integral. Both are defensible and they give different numbers with different mesh sensitivity; the exit-plane form stays valid when the domain extends past the lip into the plume, which it does.

**Quasi-1D reference** (verified end to end for r_t = 50 mm, ε = 25, γ = 1.20, Mw = 22, p₀ = 5 MPa, T₀ = 3200 K, p_a = 101325 Pa — every claimed digit reproduced): M_e = 3.9127686, ṁ = 23.1584772 kg/s via both the Vandenkerckhove Γ and ρ_e V_e A_e (agreement 6e-16), c\* = 1695.703 m/s, F = 52455.11 N, C_f = 1.335758 (closed form and momentum integral agree to 4e-16), Isp_SL = 230.971 s, Isp_vac = 318.573 s. Use this as a unit test for the reference path.

**What a correct 2D axisymmetric solver should show against quasi-1D**, so a developer can tell physics from a bug:

| | quasi-1D | correct 2D | why |
|---|---|---|---|
| ṁ | ideal | 0.3–1.0% lower | curved sonic line; discharge coefficient below 1 |
| exit Mach | uniform | centreline 0.1–0.3 above wall; area-average within 1% | radial equilibrium |
| C_f | ideal | 0.5–1.5% lower | divergence loss |
| sonic line | flat at throat | bulges downstream 0.1–0.3 r_t on the axis | standard. A flat sonic line means your solver is 1-D in disguise |

Bug indicators, not physics: ṁ more than 2% off, ṁ *above* the 1-D value, exit Mach outside ±5% area-averaged, exit-Mach non-uniformity above 0.5 in a bell (the Bézier is throwing an internal shock — check θ_n and that Q lies between N and E).

---

## 12. Acceptance ladder

Headless, exit 0/1, one number per line. T0–T5 run in under two minutes total.

| | Test | Reference | Pass |
|---|---|---|---|
| **T0** | Rasterizer area, no flow, <1 s | πR² for a disk on a cell corner | rel area error ≤ 0.5% at R ∈ {3,4,6,8,12,20}. ✓ Exact sub-cell fractions measure 1.4e-13%; binary point-sampling measures **+13.18% at R=3, +3.45% at R=8, −0.97% at R=12**. Note the sign at R=12 — a signed assertion fails |
| **T1** | Free-stream preservation, planar-uniform, 200 steps | source vanishes identically at v = 0 | max\|ρ−1\|, \|u−2\|, \|p−1\| ≤ 1e-5; max\|ρv\| ≤ 1e-6 |
| **T2** | Conservation drift, periodic, 1000 steps, f64 diagnostics | exact | ≤ 2e-6 relative, measured against the geometry-edit ledger, not against zero |
| **T3** | Order of accuracy, smooth advection, 1D periodic, one period | exact = initial condition | **unlimited slope ≥ 1.90**, limited ≥ 1.50. Two thresholds, because a limited scheme cannot hit 2.0 and "1.6" alone is ambiguous between a healthy limiter and broken reconstruction. This is the cheapest unambiguous test on the ladder |
| **T4** | Sod, 1-cell-tall strip, N = 200, t = 0.2, γ = 1.4 | ✓ p\* = 0.3031301781, u\* = 0.9274526200, ρ\*_L = 0.4263194282, ρ\*_R = 0.2655737117, shock speed 1.7521557320; positions 0.2633568 / 0.4859454 / 0.6854905 / 0.8504311 | L1(ρ) ≤ 6.0e-3. Correct 2nd order measures 2.4–4.1e-3, first order 1.32e-2 — the threshold sits in the gap. Also: shock front within ±1.5 dz, max ρ ≤ 1.001, max\|ρv\| ≤ 1e-8 (catches transverse flux leakage) |
| **T5** | Positivity, Toro test 2 and the vacuum-end nozzle | p\* = 1.894e-3 | floor activations **= 0**. Nonzero is a hard stop, not a soft failure — every downstream number becomes un-auditable |
| **T6** | Oblique shock off a 15° wedge, M = 2, γ = 1.4 | ✓ β = **45.34362°**, p₂/p₁ = 2.1946531, ρ₂/ρ₁ = 1.7289223, M₂ = 1.4457164, dβ/dθ = 1.3490216 | β within ±1.5°, fitted by least squares over x ∈ [x₀+60dz, x₀+150dz]. **Exclude the first 60 cells** — the leading edge is where a staircase wall is worst and including it fails a correct solver |
| **T7** | Cone vs Taylor–Maccoll, M = 2.35, θ_c = 10°, γ = 1.4 | ✓ β = **26.736718°**, M_surface = 2.146831, p_c/p_∞ = **1.373936** | β ±1.5°, surface p ±8%, surface M ±5%. Plus: max \|v\| in the two rows next to the axis upstream of the cone ≤ 1e-3 of freestream |
| **T8** | Nozzle vs isentropic | recomputed from `case.gas.gamma` | ṁ/ṁ_ideal ∈ [0.94, 1.00]; exit Mach area-averaged ±4%; C_f 0.975–0.995 of ideal |

**T7 has a trap.** The NASA NPARC validation archive lists p₃/p₁ = 1.4234 for this case. That value is wrong for a 10° cone — ✓ it corresponds to a **10.791°** cone (⚠ the earlier estimate of 10.90° was itself off; 10.90° gives 1.43040). The archive's own pair is internally inconsistent: its β = 27.1843° implies p = 1.4306, not 1.4234. Its *Mach* numbers match to five digits. Put this in the test comment so nobody "fixes" the value back to the archive's.

**T1 cannot catch a wrong axisymmetric source term** — the source is identically zero there. T7 is the only test that validates the axisymmetric machinery against a nontrivial exact solution. In an axisymmetric app, it is not optional.

---

## 13. Honesty rules

**Separation warning.** Show when `p_e/p_a < max(0.40, (1.88·M_e − 1)^−0.64)`.

✓ The Schmucker form was reconstructed from two OCR-garbled sources and is confirmed against NASA TM-77396 eq. (11), valid to M ≤ 5. Values: 0.522 at M = 2, 0.433 at 2.5, 0.374 at 3, 0.301 at 4, 0.256 at 5, crossing Summerfield's 0.40 at M = 2.758. It brackets the independent Zukoski criterion within 10% over M = 2–6.

⚠ Note that `max()` makes the Schmucker term dead code in the demo range: every nozzle at M_e > 2.758 has a Schmucker threshold below 0.40, so the trigger is always exactly 0.40. That is correct if the intent is "warn at whichever criterion fires first as ambient pressure rises," which it is. Keep `max()`; just do not expect Schmucker to participate at the demo settings.

Wording when it fires:

> **⚠ SEPARATED FLOW — NOT SIMULATED.** Exit pressure ratio p_e/p_a = 0.28 is below the separation threshold of 0.40. A real nozzle would separate inside the divergent section: the boundary layer would detach and a shock would stand inside the nozzle. **This simulation is inviscid — it has no boundary layer and cannot separate.** Thrust and exit-pressure readouts are not valid in this regime.

Shade the altitude slider track below the trigger and put a labeled tick at the crossing point, recomputed when the area ratio changes, so the user sees where the trustworthy range ends before dragging.

**Refuse to display:** Isp in seconds (needs real gas and chemistry — show C_f relative to the 1-D ideal instead, which is a ratio and defensible); any efficiency to better than 1%; wall heat flux, shear, skin friction, boundary-layer thickness or wall temperature (there is no boundary layer — these are undefined, not merely inaccurate); separation station as a simulation result; absolute thrust for a named real engine; anything at all while the positivity floor is **active or recently active** — any activation within the last 1500 steps blanks the report outright.

⚠ The floor rule was binary ("anything at all while the counter is nonzero") when the floor sat at 1e-8. At the raised 1e-6·p₀ floor (§5) the cold-start front of the high-pressure presets *legitimately* passes below the floor — measured: Raptor 2 ~600 activations ending by step 130, RS-25 ≤ 2.9k ending by step 220, Merlin Vac ≤ 149k ending by step 796, all engines zero activations ever after, and ambient transitions add exactly zero (re-measured 2026-08 on the configurable-domain preset grids, whose plume-core radial hold adds cells in the expansion region: counts grow — Raptor 2 ≤ 4.6k, RS-25 ≤ 17k, Merlin Vac ≤ 256k — but every window still ends by step 753 and the transition path still adds exactly zero, so the 1500-step quarantine sized off the 796 worst case stands) — so a permanently-poisoned counter would blank three of the six presets forever for a transient whose invented mass leaves the domain in milliseconds. The large graded domains' sea-level cold starts are in the same class — the startup blast expands deeper before relief in a larger domain (measured, grading bench 2026-08: Standard 551 activations, Large 9,280, both ending by step 301, zero ever after; the compact-sized domains stay clean; the benchmark asserts the startup-confined shape for every domain shortcut). The quarantine form keeps the spirit: while activations are recent the field is being invented and nothing displays; after 1500 quiet steps (≈ 2× the longest observed activation window) the report returns **with a permanent amber disclosure** of the count — the event is never hidden, and a floor that fires at steady state keeps the report blank indefinitely.

**Display with a badge:** mass flow (±(100/N_throat)% computed live from the detected throat resolution — ±5% at the reference 20 cells/r_t, matching the §8 table — amber below N_throat = 20); exit pressure and exit Mach ("area-averaged, ±4%" — the exit plane is genuinely non-uniform, so a single number is a choice — plus the first-order staircase-wall bias, ≈19% low at N_throat = 20, scaled live as 20/N_throat); thrust coefficient ("inviscid, no divergence correction applied", or apply λ and say so — pick one); shock angles ±1.5°; everything in drawn-geometry mode ("sandbox — qualitative only", since no acceptance test covers an arbitrary user drawing).

**One permanent line of screen space:** `Inviscid Euler · γ = 1.24 · no boundary layer · no chemistry · no heat transfer`.
