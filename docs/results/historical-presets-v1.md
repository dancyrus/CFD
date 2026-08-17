# Historical engine presets, v1 (1943–1961)

Eight engines spanning the V-2 to Titan I, with chamber thermochemistry computed
rather than taken from handbooks, and throat radii derived through a method that
reproduces the one published throat dimension in the set to −0.03%.

| id | preset name | year | p₀ MPa | ε | r_t mm | class | contour | default alt |
|---|---|---|---|---|---|---|---|---|
| `v2_a4` | V-2 (A-4) Model 39 | 1943 | 1.572 | 2.83 | 193.6 | LoxEthanol75 | 15° cone | 0 km |
| `wac_corporal` | WAC Corporal 38ALDW-1500 | 1945 | 2.068 | 5.00 | 27.5 | RfnaAnilineFurfuryl | 15° cone | 0 km |
| `redstone_a7` | Redstone NAA 75-110-A-7 | 1953 | 2.193 | 3.61 | **196.85** | LoxEthanol75 | 15° cone | 0 km |
| `thor_lr79` | Thor LR79-NA-7 (MB-1) | 1957 | 4.100 | 8.00 | 187.8 | LoxRp1 | 80% Rao bell | 0 km |
| `titan1_lr87` | Titan I LR87-AJ-3 (1 of 2) | 1959 | 4.000 | 8.00 | 187.7 | LoxRp1 | 80% Rao bell | 0 km |
| `titan1_lr91` | Titan I LR91-AJ-3 | 1959 | 4.500 | 25.0 | 118.9 | LoxRp1 | 75% Rao bell | **12 km** |
| `atlas_lr89` | Atlas LR89-5 booster (1 of 2) | 1960 | 3.999 | 8.00 | 198.7 | LoxRp1 | 80% Rao bell | 0 km |
| `atlas_lr105` | Atlas LR105-5 sustainer | 1960 | 4.799 | 25.0 | 120.0 | LoxRp1 | 80% Rao bell | **10 km** |

**The bell/cone transition falls inside this date range, not after it.** Only the
V-2, WAC Corporal and Redstone are cones. Rocketdyne carried bell nozzles from
Navaho Phase II/III into Atlas, Jupiter and Thor, and Rao was at Rocketdyne, so
Thor, both Atlas engines and both Titan I engines are bells. An earlier revision
of this table had all eight as cones; that was wrong on five of them.

Machine for every measured number below: `intel-xeon-4c`. Raw records:
`docs/results/historical-presets-intel-xeon-4c.json`. CEA output:
`docs/results/propellant-cea.md`.

## 1. Thermochemistry

`tools/propellant_cea.py`. Constant-enthalpy, constant-pressure equilibrium from
liquid-reactant heats of formation, on the GRI-Mech 3.0 species set with the
**high-temperature** thermo fits (`gri30_highT`). The standard `gri30`
polynomials cap at 3000 K and every LOX/RP-1 case here runs above 3450 K, so the
standard set would have been extrapolating on all five.

| engine | O/F | p₀ MPa | T₀ K | Mw | γ (frozen) | c* shifting | c* frozen |
|---|---|---|---|---|---|---|---|
| V-2 Model 39 | 1.130 | 1.572 | 2889.6 | 22.032 | 1.1946 | 1634.5 | 1612.8 |
| Redstone 75-110-A-7 | 1.324 | 2.193 | 3081.6 | 23.356 | 1.1875 | 1655.0 | 1621.1 |
| Thor LR79-NA-7 | 2.240 | 4.100 | 3495.1 | 21.946 | 1.2206 | 1804.4 | 1763.6 |
| Atlas LR89-5 booster | 2.210 | 3.999 | 3475.9 | 21.807 | 1.2215 | 1803.9 | 1763.8 |
| Atlas LR105-5 sustainer | 2.250 | 4.799 | 3518.5 | 22.026 | 1.2197 | 1806.7 | 1766.7 |
| Titan I LR87-AJ-3 | 2.200 | 4.000 | 3470.2 | 21.762 | 1.2218 | 1803.8 | 1764.0 |
| Titan I LR91-AJ-3 | 2.200 | 4.500 | 3483.1 | 21.787 | 1.2214 | 1805.4 | 1766.5 |
| WAC Corporal 38ALDW | 2.650 | 2.068 | 2988.3 | 25.620 | 1.2116 | 1539.1 | 1513.2 |

### Two characteristic velocities, and why both are carried

`c* shifting` is by equilibrium expansion to the sonic point, letting the
composition shift. It is what CEA reports and what the published throat areas
were sized against, so it is what the derivation in §2 is calibrated on.

`c* frozen` is the closed form for a calorically perfect gas at the same
(T₀, Mw, γ). It runs ~1.3% low, because holding chamber γ all the way to the
throat ignores the recombination that reheats the gas as it accelerates. It is
nonetheless the honest figure for **this** solver, which is exactly that fixed-γ
ideal gas.

Substituting one for the other would move every derived throat radius by ~1.3%,
which is 40× the gate's own margin — so the two are pinned to each other:
`rust_and_python_frozen_cstar_agree` recomputes the frozen form in Rust from the
committed (T₀, Mw, γ) and asserts it matches the tool's column to 1e-5.

### γ is the frozen chamber value

The solver runs a fixed-γ ideal gas with no species transport
(docs/physics-reference.md §2), so cp/cv of the equilibrium chamber mixture is
the correct value to hand it.

One useful consequence: the five RP-1 engines compute to γ 1.2197–1.2218, which
sits close to the 1.23–1.25 band the Rao table was digitised at — far closer than
Raptor 2 at 1.16. The two engines that fall well outside it, the V-2 and Redstone
at 1.19, are cones and never consult the table.

### The `LoxRp1` class is not re-baselined

The app's six pre-existing presets carry γ 1.24 as a per-engine literal. This
session computes 1.221 as the frozen chamber value. Both are defensible — 1.24 is
a reasonable expansion-averaged effective γ, 1.221 is the chamber value — and
changing what the F-1 and both Merlins run would re-baseline every committed
benchmark in the repo.

So it isn't changed. The six keep their literals and `propellant_class: None`;
the CEA-derived `LoxRp1` class reaches only the five new RP-1 presets.
`existing_six_presets_are_unchanged` pins all six field by field, including the
five fields this work added. Reconciling 1.24 against 1.221 is a separate item.

## 2. Throat radii, and why the obvious method fails

Only one published throat dimension exists in this engine set: the Redstone, at
15.5 in diameter.

**Rejected: A_t = ṁ·c*/p₀.** For the Redstone this back-solves to a delivered c*
of 1657.5 m/s against a computed ideal of 1655 — a c* efficiency of 100.2%. Real
engines deliver 92–96%. Something in that published quartet is off by ~5%, most
likely chamber pressure quoted at the injector face rather than as throat
stagnation, or mass flow including gas-generator flow.

Independent confirmation that the mass-flow column is the weak one: the AEHS
LR89-5 row lists 163,211 lbf against 458 lb/s, which implies an Isp of 356 s. The
same table's own Isp column says 248–282. The Redstone and Titan GLV rows are
self-consistent, so this is a bad cell rather than a systematic units error.

**Adopted: A_t = F / (C_f,ideal · λ · η · p₀)**, with published thrust at the
appropriate ambient, ideal C_f from the computed γ and area ratio, a divergence
factor λ of (1+cos 15°)/2 = 0.9830 for a cone and 0.988 for a bell, and a
residual nozzle efficiency η = 0.9877 calibrated on the Redstone.

| engine | F published | rated at | C_f ideal | λ | η | A_t m² | r_t mm |
|---|---|---|---|---|---|---|---|
| V-2 Model 39 | 244.7 kN | sea level | 1.3608 | 0.9830 | 0.9877 | 0.117797 | 193.64 |
| Redstone 75-110-A-7 | 369.1 kN | sea level | 1.4253 | 0.9830 | 0.9877 | 0.121656 | **196.79** |
| Thor LR79-NA-7 | 667.2 kN | sea level | 1.5051 | 0.9880 | 0.9877 | 0.110794 | 187.79 |
| Atlas LR89-5 booster | 726.0 kN | sea level | 1.4997 | 0.9880 | 0.9877 | 0.124056 | 198.72 |
| Atlas LR105-5 sustainer | 386.4 kN | vacuum | 1.8249 | 0.9880 | 0.9877 | 0.045215 | 119.97 |
| Titan I LR87-AJ-3 | 647.9 kN | sea level | 1.4996 | 0.9880 | 0.9877 | 0.110689 | 187.71 |
| Titan I LR91-AJ-3 | 355.9 kN | vacuum | 1.8235 | 0.9880 | 0.9877 | 0.044447 | 118.94 |
| WAC Corporal 38ALDW | 6.7 kN | sea level | 1.3953 | 0.9830 | 0.9877 | 0.002381 | 27.53 |

**Validation gate: derived Redstone r_t = 196.79 mm against published 196.85 mm,
−0.03%.** The gate asserts under 1%.

The shipped table carries the published 196.85 mm for the Redstone and the
derived value for the other seven. `derived_throat_radii_reproduce_the_published_thrust`
re-runs this derivation from each preset's own committed γ every test run and
fails if any radius has drifted more than 0.5% from what the published thrust
implies; worst observed is the WAC Corporal at 0.12%, which is decimal rounding
of the committed literal.

Confidence on the seven derived radii is roughly ±3%, driven by η resting on a
single calibration point.

### Mass flow: recorded, never asserted

The work order's constraint was to report a >10% mass-flow disagreement rather
than move a throat radius to close it. No radius was moved, and none needs to
be — but the comparison is deliberately not an assertion either, because the
published column it would compare against is the one shown above to be
unreliable. Implied mass flows at the tool's 95% c* efficiency are recorded per
preset in the results JSON.

Worth noting against the previous revision of this table: its mass-flow-derived
radii produced three engines more than 10% from published performance (V-2
+10.5%, Atlas LR89 −29.5%, WAC Corporal +61%). The thrust-derived radii here
reproduce published thrust by construction, and the two large outliers moved a
long way — the Atlas LR89 from 170 to 198.7 mm, the WAC Corporal from 35 to
27.5 mm.

## 3. Contour provenance

| preset | contour | confidence |
|---|---|---|
| V-2 | 15° cone | Confirmed. Wewerka set the half-angle; AEHS names it conical |
| WAC Corporal | 15° cone | Confirmed by date — 1945 Aerojet JATO unit |
| Redstone | 15° cone | Confirmed: "a straight-sided 15° divergent nozzle section was retained" |
| Thor LR79 | 80% Rao bell | Family confirmed; percentage is a design-class estimate |
| Atlas LR89 | 80% Rao bell | Family confirmed; percentage is a design-class estimate |
| Atlas LR105 | 80% Rao bell | Family confirmed; percentage is a design-class estimate |
| Titan I LR87 | 80% Rao bell | **Family unverified** — Aerojet, not Rocketdyne |
| Titan I LR91 | 75% Rao bell | **Family unverified.** Ablative skirt carries ε 13:1 → 25:1 |

No published wall angles exist for any of the five bells. Thor is 1957 and
Atlas D 1959, against Rao's 1958 and 1960 papers — these engines predate or are
contemporary with the charts the generator interpolates, so the wall it draws is
of the right family but is not the engine's own contour. Every bell preset's
tooltip says so.

`cone_half_angle_deg` is now per-preset rather than a hard-coded 15°. The contour
audit found no source for that constant, and handbook practice puts the divergent
half-angle anywhere from 12 to 18°. Three presets here genuinely are 15°, which
is three data points and not a justification for a constant. The six pre-existing
presets are all bells, so the field is inert for them.

## 4. Separation and the default altitude

Quasi-1D p_e/p_a against the 0.40 Summerfield threshold:

| preset | M_e | p_e kPa | p_e/p_a at SL | crossing | default alt | p_e/p_a there |
|---|---|---|---|---|---|---|
| V-2 | 2.345 | 113.3 | 1.118 | — | 0 km | 1.118 |
| WAC Corporal | 2.806 | 64.4 | 0.635 | — | 0 km | 0.635 |
| Redstone | 2.524 | 112.9 | 1.114 | — | 0 km | 1.114 |
| Thor LR79 | 3.174 | 65.6 | 0.648 | — | 0 km | 0.648 |
| Titan I LR87 | 3.177 | 63.8 | 0.630 | — | 0 km | 0.630 |
| Titan I LR91 | 4.014 | 15.9 | **0.157** | **7.2 km** | **12 km** | 0.821 |
| Atlas LR89 | 3.176 | 63.9 | 0.630 | — | 0 km | 0.630 |
| Atlas LR105 | 4.006 | 17.0 | **0.168** | **6.7 km** | **10 km** | 0.644 |

The two altitude-optimised stages genuinely separate at sea level. Opening them
at 0 km would raise the separation warning the instant the preset is clicked,
about nothing the user did — and a warning that fires on arrival is one people
learn to dismiss. The shipped defaults clear their own computed crossings by
4.8 km and 3.3 km.

The test asserts this in **both** directions, plus the margin. Every zero-default
preset must *not* separate at sea level, or it would need a default of its own;
the two non-zero-default presets *must* separate at sea level, or the default
would be decoration; and each non-zero default must clear its own bisected
crossing by at least 2 km. A field checked only in the passing direction can be
all zeros and still pass.

**The V-2 and Redstone are slightly under-expanded at sea level** (1.118 and
1.114). That is the design intent for a surface-launched vehicle of the period,
and it makes them the best teaching cases in the library: the altitude slider
crosses perfect expansion right at the bottom of its range rather than somewhere
in the middle.

## 5. Rao table clamp, both ends

The digitised table covers ε = 4–100 and `rao_angles` clamps at both ends rather
than extrapolating. The area-ratio slider bottoms out at ε = 2, so all five
historical bells can be dragged *below* the table — the mirror of the ε > 100
case that Merlin Vac (ε 165) already exercises.

The app's disclosure fires on `!(4.0..=100.0).contains(ε)`, which already covered
both ends. What this change adds is the guarantee: the range is now one shared
constant (`case::RAO_TABLE_EPS`) used by both the UI condition and
`rao_table_clamps_and_flags_symmetrically_at_both_ends`, so the clamp and the
disclosure cannot drift apart. The test also verifies the two end rows are
genuinely distinct, so "clamps to the end row" is a real constraint rather than a
tautology about a flat table, and that a bell at ε = 3 still generates rather
than falling back to a cone — a fallback would bypass the disclosure entirely and
put "15° cone" in the status line instead.

## 6. Run time to convergence

Measured, not projected: throughput over 250 steps after a 50-step warm-up, then
the case run to the §9 visual-steady step count with the wall clock taken.

| preset | ε | grid | cells | steps/s | steps | **wall time** | × Merlin 1D |
|---|---|---|---|---|---|---|---|
| V-2 | 2.83 | 105 × 188 | 19,740 | 289.5 | 4,668 | **17.9 s** | 0.26 |
| Redstone | 3.61 | 113 × 188 | 21,244 | 275.4 | 4,983 | **20.3 s** | 0.30 |
| WAC Corporal | 5.00 | 124 × 201 | 24,924 | 235.6 | 5,469 | **25.1 s** | 0.38 |
| Titan I LR87 | 8.00 | 137 × 234 | 32,058 | 208.4 | 6,325 | **35.2 s** | 0.57 |
| Thor LR79 | 8.00 | 137 × 234 | 32,058 | 202.8 | 6,325 | **35.4 s** | 0.57 |
| Atlas LR89 | 8.00 | 137 × 234 | 32,058 | 193.2 | 6,325 | **35.5 s** | 0.57 |
| Titan I LR91 | 25.0 | 197 × 325 | 64,025 | 109.5 | 9,466 | **93.2 s** | 1.70 |
| Atlas LR105 | 25.0 | 201 × 325 | 65,325 | 108.3 | 9,466 | **94.3 s** | 1.73 |

The cheapest cases in the library, as expected: 18–94 s against the demo case's
Large domain at 804 s and Merlin Vac's ~12× Merlin 1D. ε runs 2.83–25 here
against 69 and 165 for the existing worst cases, and `preset_domain` sizes the
domain off ε, so a short nozzle buys a small domain as well as a short one.

Run-to-run spread on this machine is about 2%, so treat the last digit as noise.
Cell and step counts are exact — they are geometry, not timing.

### Positivity floors

| preset | activations | last at step | of |
|---|---|---|---|
| WAC Corporal | 4,859 | 132 | 5,469 |
| Titan I LR91 | 12,802 | 236 | 9,466 |

The other six never touch the floor. Both cases are the §13 cold-start class —
confined to the startup front and silent afterwards — and the benchmark asserts
the last activation lands in the first half of the run. Steady-state floor
contact would be a hard failure: cumulative activations blank every readout under
the product rule.

Note that the Atlas LR105 now shows **zero** activations where the previous
revision recorded 6,868. Its default altitude dropped from 25 km to 10 km, so the
startup front expands into a much thicker ambient.

### Solved mass flow vs quasi-1D

Recorded per preset in the results JSON. The solved figures sit below the
quasi-1D ideal for the same throat and frozen c* by a roughly uniform margin
across all eight — the §8 1/N_throat systematic error at 20 cells/r_t, which the
report's own N_throat badge exists to flag. It is a property of the grid, not a
verdict on any preset's throat radius.

## 7. Open items

1. **`LoxRp1` γ, 1.24 vs 1.221.** Deliberately not reconciled here; see §1.
2. **Titan I contour family unverified.** Both LR87 and LR91 ship as bells on
   judgement, not evidence — Aerojet, not Rocketdyne, so the Navaho bell lineage
   does not carry over. Resolvable from museum hardware at Kirtland AFB or the
   Titan Missile Museum, both of which have the nozzles visible.
3. **V-2 area ratio contested.** Sutton via AEHS gives 2.83; museum-hardware
   dimensions quoted elsewhere imply ~3.34, an 18% difference worth roughly 4% on
   exit Mach. 2.83 is used because it is the only figure traced to a source.
4. **WAC Corporal is a reconstruction.** Chamber pressure and area ratio are both
   unpublished. Thrust, burn time, propellants and the air-pressurised feed are
   confirmed. Shipped labelled.
5. **η = 0.9877 rests on one point.** Every derived throat radius inherits the
   Redstone calibration. A second published throat dimension anywhere in the set
   would turn a calibration into a check.
6. **Bell percentages are design-class estimates.** No published wall angles
   exist for any of the five bells.
