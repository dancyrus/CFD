# Session C — `cfd-geom`

**Budget: 70 minutes.** You have the most self-contained job in the build and no dependency on the solver. Finish early and your session becomes a second pair of hands for whoever is behind.

## Read first

`docs/physics-reference.md` §8 and §10, `docs/contract.md`, `cfd-contract/src/lib.rs`.

## Rules

- You own **`cfd-geom/` and nothing else.**
- **No egui dependency.** The editor here is a data model, not a widget. Session D wraps it.
- Never modify `cfd-contract/`. If it is wrong, print `CONTRACT CHANGE REQUEST:` with the diff and stop.
- Lengths are SI (metres) in `NozzleSpec` and `WallProfile`; `rasterize` converts to the non-dimensional grid through `RefScales`.

## What you build

### 1. Conical contour

Converging section: contraction ratio 4 (chamber radius = 2 r_t), converging half-angle 30°, upstream throat arc 1.5 r_t, chamber straight length 2 r_t. Downstream throat arc 0.382 r_t.

```
throat arc, parameterized by local wall angle phi in [0, alpha]:
    z(phi) = R_cd * sin(phi)
    r(phi) = r_t + R_cd * (1 - cos(phi))
tangency point:
    zA = R_cd * sin(alpha);   rA = r_t + R_cd * (1 - cos(alpha))
cone:
    r(z) = rA + (z - zA) * tan(alpha),   zA <= z <= L_n
L_n = ( r_t*(sqrt(eps) - 1) + R_cd*(sec(alpha) - 1) ) / tan(alpha)
```

That closed form for `L_n` is exact — the identity was verified symbolically, worst numeric residual 4.2e-15 over 20,000 random parameter draws. Parameterizing the arc by wall angle rather than by a circle parameter makes tangency automatic.

### 2. Parabolic bell (Rao approximation)

Same converging section and 1.5 r_t arc. The downstream arc is 0.382 r_t and runs to wall angle `theta_n`, not to 15°. Then a quadratic Bézier.

```
L_c15 = ( r_t*(sqrt(eps)-1) + 0.382*r_t*(sec(15deg)-1) ) / tan(15deg)
L_n   = bell_percent * L_c15

N = ( 0.382*r_t*sin(theta_n),  r_t + 0.382*r_t*(1 - cos(theta_n)) )
E = ( L_n,  r_t*sqrt(eps) )
m1 = tan(theta_n);  m2 = tan(theta_e)
C1 = N_r - m1*N_z;  C2 = E_r - m2*E_z
Q  = ( (C2-C1)/(m1-m2),  (m1*C2 - m2*C1)/(m1-m2) )
P(t) = (1-t)^2 * N + 2t(1-t) * Q + t^2 * E
```

Verified for r_t = 50 mm, ε = 25, 80% bell: wall angle is exactly `theta_n` at t = 0 and exactly `theta_e` at t = 1 (error 0.00e+00 both), `N_z < Q_z < E_z`, and the curve is monotone in both z and r.

**Two guards the code must have.** Assert `theta_n > theta_e`, or `m1 - m2 = 0` and you divide by zero. Assert `N_z < Q_z < E_z` — if `bell_percent` is too small for the area ratio, Q falls outside and the contour turns back on itself.

`t` is not proportional to z. To sample uniformly in z, sample t densely and resample.

**Two circulating definitions of `L_c15` differ by 0.337%.** The form above (Huzel–Huang) includes the throat-arc term; Aspirespace and the widely-copied `bell_nozzle.py` drop it. Use the form above and put a comment saying so, or "80% bell" is ambiguous.

### 3. `rao_angles(area_ratio, bell_percent) -> (theta_n, theta_e)`

Digitized Rao table, interpolated **log-linearly in ε** and linearly in bell percent. There is no published polynomial fit; every implementation uses a table.

Anchors (degrees), abscissa = ε = [4, 5, 10, 20, 30, 40, 50, 100]:

```
tn_60 = [26.5, 28.0, 32.0, 35.0, 36.2, 37.1, 37.5, 40.0]
tn_80 = [21.5, 23.0, 26.3, 28.8, 30.0, 31.0, 31.5, 33.5]
tn_90 = [20.0, 21.0, 24.0, 27.0, 28.5, 29.5, 30.2, 32.0]
te_60 = [20.5, 20.5, 16.0, 14.5, 14.0, 13.5, 13.0, 11.2]
te_80 = [14.0, 13.0, 11.0,  9.0,  8.5,  8.0,  7.5,  7.0]
te_90 = [11.5, 10.5,  8.0,  7.0,  6.5,  6.0,  6.0,  6.0]
```

Clamp `bell_percent` to [0.6, 0.9] and ε to [4, 100]. Do not extrapolate to 100% bells; the published data does not support it.

**The 60% `tn` row as commonly circulated is non-monotonic** (37.1 at ε=40 → 35.0 at ε=50, a digitization typo). The value above is corrected to 37.5. It does not affect an 80% bell, but if the UI exposes bell percent, an uncorrected 60% row produces a nozzle whose divergence angle decreases with area ratio.

Test: at ε = 25, 80% bell, log-linear interpolation must give **29.4604° / 8.7248°**. Linear-in-ε gives 29.400 / 8.750, which is how you tell the two apart.

Note in a comment that the table was digitized at a specific γ (around 1.23–1.25), or the divergence factor is quietly inconsistent later.

### 4. `rasterize` — exact sub-cell area fractions

This is the highest-risk function in your crate, and the failure mode is silent.

Point sampling (`if cell_center_inside { solid = 1 }`) gives **+13.18% area error at radius 3 cells, +3.45% at 8, −0.97% at 12** — and it does not converge monotonically, it oscillates. Exact sub-cell area fractions measure 1.4e-13%. Every geometry-dependent number downstream inherits whichever you choose, and the picture looks normal either way.

Compute the exact fraction of each cell lying outside the wall. Sub-sampling at 4 samples per edge is the fallback if an analytic intersection is awkward, but analytic is better and the test threshold assumes something close to it.

**Test T0, run first, no flow, under a second:** rasterize a disk of radius R centred on a cell corner, sum fraction × cell area, compare to πR². Relative error ≤ 0.5% at R ∈ {3, 4, 6, 8, 12, 20}. Note the error is **signed** — negative at R = 12 — so a signed assertion will fail spuriously; take the absolute value.

Second assertion in the same test: for the parametric nozzle at five slider settings, the throat area recovered from the solid field is within 0.5% of the analytic throat area.

### 5. Throat quantization

Choose `r_t` (or `dr`) so the throat lands exactly on a cell face. The ±1/N_throat mass-flow error is one geometric quantization of `r_t` against `dr`; aligning them collapses the systematic term to the second-order arc-curvature effect. About ten lines.

**Parametric nozzle only.** Do not apply it to drawn geometry, which has no theoretical thrust to match.

### 6. Drawn-polyline path

A user-drawn polyline produces the **identical `SolidField` type** through the same `rasterize`. That is the whole architectural point: the solver cannot tell a drawn wall from a generated one, so one solver serves both. `WallProfile` is simultaneously the generated contour and the editable representation — dragging a control point edits `points` directly.

### 7. Editor data model

Control points, selection, hit testing, world/screen transform, insert, drag, remove. Plain Rust. Session D wraps this in egui and consumes it behind a trait, so keep the API in `docs/contract.md` exactly.

Validation on `to_profile`: monotone in z, all r > 0, no self-intersection, `throat_index` in range.

## Done when

`cargo test -p cfd-geom` is green, including T0 and the Rao interpolation test.

## Known traps

- Point-sampled rasterization. Looks fine on screen, poisons every number.
- Signed area error in the T0 assertion.
- The `L_c15` ambiguity.
- Dropping the throat-arc term from the conical length (0.34% short).
- Adding an egui dependency "just for the editor." It costs every other worktree a rebuild.
