# Propellant thermochemistry for the historical engine presets

Verbatim output of `tools/propellant_cea.py`. Regenerate with:

```
pip install cantera scipy
python3 tools/propellant_cea.py
```

Constant-enthalpy, constant-pressure equilibrium from liquid-reactant heats of
formation, on the GRI-Mech 3.0 species set with the **high-temperature** thermo
fits (`gri30_highT.yaml`). The standard `gri30` polynomials cap at 3000 K and
every LOX/RP-1 case here runs above 3450 K, so the standard set would be
extrapolating on all five.

`gamma` is the FROZEN chamber value (cp/cv of the equilibrium mixture), because
the solver runs a fixed-gamma ideal gas. `c*` is by shifting-equilibrium
expansion to the sonic point, NOT the frozen-gamma closed form — the closed form
reads ~1.3% low because it holds chamber gamma all the way to the throat and so
ignores the recombination that reheats the gas as it accelerates. The throat
derivation, and therefore the validation gate, depends on the shifting figure.

`r_t` is derived as `A_t = F / (Cf_ideal · lambda · eta · p0)` from published
thrust — not from mass flow and c*, which the AEHS tables do not support (see
`historical-presets-v1.md` §2).

## Output

```
case                        p0 MPa   T0 K     Mw  gamma  c* m/s  Cf id  r_t mm   M_e  pe/pa SL
----------------------------------------------------------------------------------------------
V-2 Model 39                  1.57   2890  22.03  1.195    1635  1.361   193.6  2.34     1.118
Redstone 75-110-A-7           2.19   3082  23.36  1.187    1655  1.425   196.8  2.52     1.114
Thor LR79-NA-7                4.10   3495  21.95  1.221    1804  1.505   187.8  3.17     0.648
Atlas LR89-5 booster          4.00   3476  21.81  1.222    1804  1.500   198.7  3.18     0.630
Atlas LR105-5 sustainer       4.80   3519  22.03  1.220    1807  1.825   120.0  4.01     0.168
Titan I LR87-AJ-3             4.00   3470  21.76  1.222    1804  1.500   187.7  3.18     0.630
Titan I LR91-AJ-3             4.50   3483  21.79  1.221    1805  1.823   118.9  4.01     0.157
WAC Corporal 38ALDW           2.07   2988  25.62  1.212    1539  1.395    27.5  2.81     0.635

VALIDATION GATE - Redstone published throat (15.5 in diameter)
  published r_t = 196.85 mm
  derived  r_t = 196.79 mm   error -0.03%
  GATE (|error| < 1%): PASS

Altitude needed to bring pe/pa above the 0.40 separation threshold:
  Atlas LR105-5 sustainer    pe/pa(SL) = 0.168 -> needs 6.7 km
  Titan I LR91-AJ-3          pe/pa(SL) = 0.157 -> needs 7.2 km
```

## Full precision

The table above is rounded for reading; these are the values the preset table
carries.

| case | T0 K | Mw g/mol | gamma (frozen) | c* m/s (shifting) | c* m/s (frozen form) | r_t mm | Cf ideal | M_e |
|---|---|---|---|---|---|---|---|---|
| V-2 Model 39 | 2889.6492 | 22.03174 | 1.194595 | 1634.501 | 1612.833 | 193.6384 | 1.3608 | 2.3446 |
| Redstone 75-110-A-7 | 3081.5671 | 23.35647 | 1.187463 | 1654.974 | 1621.099 | 196.7852 | 1.4253 | 2.5245 |
| Thor LR79-NA-7 | 3495.0930 | 21.94603 | 1.220567 | 1804.434 | 1763.565 | 187.7948 | 1.5051 | 3.1740 |
| Atlas LR89-5 booster | 3475.9430 | 21.80683 | 1.221536 | 1803.934 | 1763.830 | 198.7164 | 1.4997 | 3.1764 |
| Atlas LR105-5 sustainer | 3518.5008 | 22.02637 | 1.219702 | 1806.722 | 1766.679 | 119.9678 | 1.8249 | 4.0055 |
| Titan I LR87-AJ-3 | 3470.2180 | 21.76167 | 1.221839 | 1803.813 | 1764.048 | 187.7055 | 1.4996 | 3.1772 |
| Titan I LR91-AJ-3 | 3483.1491 | 21.78713 | 1.221424 | 1805.410 | 1766.514 | 118.9445 | 1.8235 | 4.0137 |
| WAC Corporal 38ALDW | 2988.2979 | 25.62019 | 1.211618 | 1539.056 | 1513.231 | 27.5308 | 1.3953 | 2.8060 |
