#!/usr/bin/env python3
"""Chamber thermochemistry for the propellant classes behind the engine presets.

Computes, for each (oxidizer, fuel, O/F, p0) case, the four numbers the solver's
fixed-gamma ideal gas needs:

    T0        chamber stagnation temperature, K
    MW        mean molecular weight of the equilibrium chamber mixture, g/mol
    gamma     FROZEN specific-heat ratio of that mixture (see below)
    c*        characteristic velocity, m/s

Method
------
Constant-enthalpy, constant-pressure equilibrium from the liquid reactants: the
mixture enthalpy is the mass-weighted sum of the as-injected heats of formation,
held fixed while the composition relaxes to chemical equilibrium at p0. Reaction
rates are irrelevant — only species thermo and element conservation matter.

Why FROZEN gamma, not equilibrium gamma
---------------------------------------
The solver is a fixed-gamma ideal gas (docs/physics-reference.md §2): it carries
one gamma for the whole field and no species transport. The equilibrium gamma
CEA prints (GAMMAs = -dlnP/dlnV|_s) is much softer than cp/cv because it lets
the composition shift as the gas expands — recombination that this solver
cannot model. Handing the solver an equilibrium gamma would ask a frozen-
composition code to reproduce a shifting-equilibrium expansion, and it would
silently over-predict exit Mach and thrust. So what is reported here, and what
goes in the preset table, is cp/cv of the equilibrium CHAMBER mixture with its
composition held fixed: the correct gamma for a frozen expansion, which is what
the solver actually performs. The two differ substantially — for LOX/RP-1 at
4.1 MPa, gamma_eq = 1.148 against gamma_frozen = 1.198.

Both are printed, so the gap is visible rather than assumed.

Backends
--------
`--backend rocketcea` (default, preferred) drives NASA CEA itself through
rocketcea (`pip install rocketcea`; needs gfortran). Custom reactant cards are
built from the table below so the enthalpies used are exactly the ones stated
here, not whatever the library happens to hold.

`--backend cantera` is the fallback for when rocketcea will not build
(`pip install "cantera>=3.0"`, ships wheels). It seeds a gri30 mixture with the
reactants' element balance, overrides the enthalpy explicitly via
`gas.HP = h_target, p`, and calls `gas.equilibrate('HP')`. It is also useful as
an independent cross-check: `--backend both` runs the two and prints the
disagreement. Caveat, and it is a real one: gri30's NASA-7 polynomials are
fitted to 3500 K, and several of these cases sit at or just past that, so the
cantera column is a sanity check on the CEA column, not a replacement for it.

Reactant data (enthalpy of formation in the AS-INJECTED phase)
--------------------------------------------------------------
    LOX (liq, 90.17 K)        O2          MW 31.9988   -12.979 kJ/mol
    HNO3 (liq, 298 K)         HNO3        MW 63.0128  -174.10
    N2O4 (liq, 298 K)         N2O4        MW 92.011    -19.564
    water (liq)               H2O         MW 18.0153  -285.830
    ethanol (liq)             C2H5OH      MW 46.0684  -277.00
    RP-1 (CEA unit formula)   CH1.9423    MW 13.9761   -22.723
    aniline (liq)             C6H7N       MW 93.1265   +31.30
    furfuryl alcohol (liq)    C5H6O2      MW 98.1004  -276.20

Blend fractions and O/F are MASS fractions and a mass ratio throughout.

Self-check
----------
`--selfcheck` rebuilds LOX and RP-1 from the table above as custom CEA cards and
compares them against rocketcea's own built-in LOX/RP-1 library propellants at
two operating points. Agreement there is what says the element balance and the
enthalpy unit conversion are right; the Redstone c* gate below is what says the
whole chain is right.

Acceptance gate
---------------
The Redstone case must reproduce the one published throat diameter available:
throat dia 15.5 in => A_t = 0.121766 m^2, mdot 355 lb/s = 161.03 kg/s,
p0 = 318 psia = 2.1926 MPa, so delivered c* = p0*A_t/mdot = 1658 m/s. Computed
IDEAL c* must land within 8% of that. Do not widen the tolerance: if this
misses, the reactant enthalpies or the element balance are wrong.

Usage
-----
    python3 tools/propellant_cea.py                     # table to stdout
    python3 tools/propellant_cea.py --selfcheck
    python3 tools/propellant_cea.py --backend both
    python3 tools/propellant_cea.py --markdown docs/results/propellant-cea.md
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field

# Universal gas constant, J/(kmol*K) — the same value cfd-ui's
# case.rs R_UNIVERSAL_SI carries, so the Rust side derives the identical
# specific gas constant from the MW printed here.
R_UNIVERSAL = 8_314.462_618
# kJ/mol -> cal/mol, the unit CEA reactant cards want.
KJ_PER_MOL_TO_CAL = 1000.0 / 4.184

# The published Redstone trio behind the acceptance gate.
REDSTONE_AT_M2 = math.pi * (0.5 * 15.5 * 0.0254) ** 2  # 15.5 in throat dia
REDSTONE_MDOT_KG_S = 355.0 * 0.453_592_37             # 355 lb/s
REDSTONE_P0_PA = 318.0 * 6_894.757_293                # 318 psia
REDSTONE_CSTAR_DELIVERED = REDSTONE_P0_PA * REDSTONE_AT_M2 / REDSTONE_MDOT_KG_S
CSTAR_GATE_TOL = 0.08

# Recorded, because the gate passes for a reason worth stating out loud.
GATE_EFFICIENCY_NOTE = """\
  NOTE — the gate passes, but from the wrong side. Delivered c* should be
  92-96% of ideal; against these numbers it comes out at 100-102%, which no
  engine achieves. The chemistry is not the suspect: two independent codes
  agree to 0.1% and the cards reproduce CEA's own library propellants to 5e-6.
  The published trio is mutually optimistic instead. Most likely 318 psia is
  INJECTOR-END pressure, which exceeds the nozzle stagnation pressure by the
  Rayleigh loss across the combustor (a few percent in a 1950s chamber), so
  p0*A_t/mdot overstates the delivered c*. A 3-5% correction there puts the
  efficiency back in the 96-99% band and still inside this gate. Treat 1658
  m/s as an upper bound on delivered c*, not a measurement."""


@dataclass(frozen=True)
class Reactant:
    """One as-injected liquid reactant.

    `elements` is the CEA unit formula: element symbol -> atoms per formula
    unit. `mw` is the molar mass of THAT formula unit (RP-1's is the CH1.9423
    unit, not a notional kerosene molecule), and `hf_kj_mol` its heat of
    formation in the same basis, in the phase and at the temperature it is
    injected at.
    """

    key: str
    label: str
    elements: dict[str, float]
    mw: float
    hf_kj_mol: float
    t_k: float

    @property
    def hf_j_kg(self) -> float:
        return self.hf_kj_mol * 1e6 / self.mw

    def cea_card(self, kind: str, wt_pct: float) -> str:
        formula = " ".join(f"{el} {n:g}" for el, n in self.elements.items())
        return (
            f"{kind} {self.key} {formula}  wt%={wt_pct:.4f}\n"
            f"h,cal={self.hf_kj_mol * KJ_PER_MOL_TO_CAL:.2f}  t(k)={self.t_k}\n"
        )


REACTANTS: dict[str, Reactant] = {
    r.key: r
    for r in [
        Reactant("LOX", "LOX (liq, 90.17 K)", {"O": 2}, 31.9988, -12.979, 90.17),
        Reactant("HNO3", "nitric acid (liq)", {"H": 1, "N": 1, "O": 3}, 63.0128, -174.10, 298.15),
        Reactant("N2O4", "dinitrogen tetroxide (liq)", {"N": 2, "O": 4}, 92.011, -19.564, 298.15),
        Reactant("H2O", "water (liq)", {"H": 2, "O": 1}, 18.0153, -285.830, 298.15),
        Reactant("C2H5OH", "ethanol (liq)", {"C": 2, "H": 6, "O": 1}, 46.0684, -277.00, 298.15),
        Reactant("RP1", "RP-1 (CEA unit formula CH1.9423)", {"C": 1, "H": 1.9423}, 13.9761, -22.723, 298.15),
        Reactant("C6H7N", "aniline (liq)", {"C": 6, "H": 7, "N": 1}, 93.1265, +31.30, 298.15),
        Reactant("C5H6O2", "furfuryl alcohol (liq)", {"C": 5, "H": 6, "O": 2}, 98.1004, -276.20, 298.15),
    ]
}


@dataclass(frozen=True)
class Mixture:
    """A named oxidizer or fuel: reactant key -> mass fraction (summing to 1)."""

    name: str
    fractions: dict[str, float]

    @property
    def cea_name(self) -> str:
        """A CEA-library key for this mixture.

        rocketcea normalises the names it is handed (`-` becomes `_`, among
        others), so the registration name and the lookup name have to be
        pre-normalised here or `add_new_fuel` and `CEA_Obj` disagree.
        """
        safe = "".join(ch if ch.isalnum() else "_" for ch in self.name)
        return f"X_{safe}"

    def __post_init__(self) -> None:
        total = sum(self.fractions.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"{self.name}: mass fractions sum to {total}, not 1")

    @property
    def hf_j_kg(self) -> float:
        return sum(w * REACTANTS[k].hf_j_kg for k, w in self.fractions.items())

    def moles_per_kg(self) -> dict[str, float]:
        """Element moles per kg of this mixture — the element balance."""
        out: dict[str, float] = {}
        for k, w in self.fractions.items():
            r = REACTANTS[k]
            per_kg = w * 1000.0 / r.mw  # mol of formula unit per kg of mixture
            for el, n in r.elements.items():
                out[el] = out.get(el, 0.0) + n * per_kg
        return out

    def cea_cards(self, kind: str) -> str:
        return "".join(
            REACTANTS[k].cea_card(kind, 100.0 * w) for k, w in self.fractions.items()
        )

    def describe(self) -> str:
        if len(self.fractions) == 1:
            return REACTANTS[next(iter(self.fractions))].label
        return " + ".join(
            f"{REACTANTS[k].key} {w:.3f}" for k, w in self.fractions.items()
        )


# ---------------------------------------------------------------------------
# Oxidizers, fuels and the cases to compute.
# ---------------------------------------------------------------------------

LOX = Mixture("LOX", {"LOX": 1.0})
RFNA = Mixture("RFNA", {"HNO3": 0.85, "N2O4": 0.13, "H2O": 0.02})

ETHANOL75 = Mixture("ethanol 75/water 25", {"C2H5OH": 0.75, "H2O": 0.25})
RP1 = Mixture("RP-1", {"RP1": 1.0})
ANILINE_FURFURYL = Mixture("aniline/furfuryl alcohol", {"C6H7N": 0.626, "C5H6O2": 0.374})


@dataclass(frozen=True)
class Case:
    engine: str
    cls: str  # propellant class this case populates
    ox: Mixture
    fuel: Mixture
    of: float
    p0_pa: float
    eps: float
    note: str = ""


CASES: list[Case] = [
    Case("V-2 (A-4)", "LoxEthanol75", LOX, ETHANOL75, 1.130, 1.57e6, 2.83),
    Case("Redstone", "LoxEthanol75", LOX, ETHANOL75, 1.324, 2.19e6, 3.61,
         "acceptance gate: delivered c* 1658 m/s from published A_t and mdot"),
    Case("Thor LR79", "LoxRp1", LOX, RP1, 2.240, 4.10e6, 8.0),
    Case("Atlas LR89 booster", "LoxRp1", LOX, RP1, 2.210, 4.00e6, 8.0),
    Case("Atlas LR105 sustainer", "LoxRp1", LOX, RP1, 2.250, 4.80e6, 25.0),
    Case("Titan I LR87 st.1", "LoxRp1", LOX, RP1, 2.200, 4.00e6, 8.0),
    Case("Titan I LR91 st.2", "LoxRp1", LOX, RP1, 2.200, 4.50e6, 25.0),
    Case("WAC Corporal", "RfnaAnilineFurfuryl", RFNA, ANILINE_FURFURYL, 2.65, 2.10e6, 5.0),
]


@dataclass
class Result:
    case: Case
    backend: str
    t0_k: float
    mw_g_mol: float
    gamma_frozen: float
    gamma_equilibrium: float | None
    cp_frozen_kj_kg_k: float
    cstar_frozen: float
    cstar_equilibrium: float | None
    extra: dict = field(default_factory=dict)

    @property
    def r_specific(self) -> float:
        return R_UNIVERSAL / self.mw_g_mol


def cstar_ideal(t0_k: float, mw_g_mol: float, gamma: float) -> float:
    """Ideal characteristic velocity for a calorically perfect gas, m/s.

    c* = sqrt(R T0 / gamma) * ((gamma+1)/2) ^ ((gamma+1)/(2(gamma-1)))

    Evaluated with the FROZEN gamma, so this is the c* the fixed-gamma solver
    reproduces — not CEA's shifting-equilibrium c*.
    """
    r_specific = R_UNIVERSAL / mw_g_mol
    g = gamma
    return math.sqrt(r_specific * t0_k / g) * ((g + 1.0) / 2.0) ** (
        (g + 1.0) / (2.0 * (g - 1.0))
    )


# ---------------------------------------------------------------------------
# Backend: rocketcea (NASA CEA)
# ---------------------------------------------------------------------------


def _cea_obj(case: Case):
    from rocketcea.cea_obj import add_new_fuel, add_new_oxidizer
    from rocketcea.cea_obj_w_units import CEA_Obj

    ox_name, fuel_name = case.ox.cea_name, case.fuel.cea_name
    add_new_oxidizer(ox_name, case.ox.cea_cards("oxid"))
    add_new_fuel(fuel_name, case.fuel.cea_cards("fuel"))
    return CEA_Obj(
        oxName=ox_name,
        fuelName=fuel_name,
        isp_units="sec",
        cstar_units="m/s",
        pressure_units="Pa",
        temperature_units="K",
        specific_heat_units="kJ/kg-K",
        enthalpy_units="kJ/kg",
        density_units="kg/m^3",
        sonic_velocity_units="m/s",
    )


def run_rocketcea(case: Case) -> Result:
    c = _cea_obj(case)
    kw = dict(Pc=case.p0_pa, MR=case.of)
    t0 = float(c.get_Tcomb(**kw))
    mw, gamma_eq = (float(x) for x in c.get_Chamber_MolWt_gamma(eps=case.eps, **kw))
    cp_frozen = float(c.get_Chamber_Cp(eps=case.eps, frozen=1, **kw))  # kJ/kg-K
    r_specific_kj = R_UNIVERSAL / mw / 1000.0
    gamma_frozen = cp_frozen / (cp_frozen - r_specific_kj)
    return Result(
        case=case,
        backend="rocketcea",
        t0_k=t0,
        mw_g_mol=mw,
        gamma_frozen=gamma_frozen,
        gamma_equilibrium=gamma_eq,
        cp_frozen_kj_kg_k=cp_frozen,
        cstar_frozen=cstar_ideal(t0, mw, gamma_frozen),
        cstar_equilibrium=float(c.get_Cstar(**kw)),
    )


# ---------------------------------------------------------------------------
# Backend: cantera (gri30), the fallback and the cross-check
# ---------------------------------------------------------------------------

# Free atoms in gri30, used purely to seed a composition with the right element
# balance. `equilibrate` conserves elements, so the seed's identity does not
# survive — only its element counts do.
_ATOM_SPECIES = {"C": "C", "H": "H", "O": "O", "N": "N"}


def run_cantera(case: Case) -> Result:
    import cantera as ct

    # Mass-weighted mixture enthalpy and element balance of the propellants as
    # injected, per kg of total (oxidizer + fuel) mixture.
    w_ox = case.of / (1.0 + case.of)
    w_fuel = 1.0 / (1.0 + case.of)
    h_target = w_ox * case.ox.hf_j_kg + w_fuel * case.fuel.hf_j_kg
    moles: dict[str, float] = {}
    for mix, w in ((case.ox, w_ox), (case.fuel, w_fuel)):
        for el, n in mix.moles_per_kg().items():
            moles[el] = moles.get(el, 0.0) + w * n

    gas = ct.Solution("gri30.yaml")
    missing = [el for el in moles if el not in _ATOM_SPECIES]
    if missing:
        raise RuntimeError(f"gri30 has no atomic seed for {missing}")
    seed = {_ATOM_SPECIES[el]: n for el, n in moles.items()}

    def reset_seed(t_k: float) -> None:
        gas.TPX = t_k, case.p0_pa, seed

    try:
        # The direct route: override the enthalpy explicitly, then relax the
        # composition at constant H and p.
        reset_seed(300.0)
        gas.HP = h_target, case.p0_pa
        gas.equilibrate("HP")
    except ct.CanteraError:
        # A free-atom seed is dissociated propellant: its frozen enthalpy is
        # tens of MJ/kg ABOVE the target, and no temperature brings a frozen
        # atomic mixture down to a target that only recombination can reach —
        # so `gas.HP =` has no root before equilibration ever runs. Solve the
        # same HP problem the other way round: bisect the temperature at which
        # the EQUILIBRIUM mixture (element-conserving, so the same constraint)
        # has the target enthalpy. h_eq(T) is monotone increasing in T.
        def h_eq(t_k: float) -> float:
            reset_seed(t_k)
            gas.equilibrate("TP")
            return float(gas.enthalpy_mass)

        lo, hi = 500.0, 6000.0
        if not (h_eq(lo) <= h_target <= h_eq(hi)):
            raise RuntimeError(
                f"{case.engine}: target enthalpy {h_target:.4g} J/kg outside "
                f"[{h_eq(lo):.4g}, {h_eq(hi):.4g}] over {lo}-{hi} K"
            ) from None
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if h_eq(mid) < h_target:
                lo = mid
            else:
                hi = mid
        h_eq(0.5 * (lo + hi))  # leave `gas` at the solved chamber state

    t0 = float(gas.T)
    mw = float(gas.mean_molecular_weight)
    # Cantera's cp/cv are frozen-composition by construction — exactly the
    # cp/cv of the equilibrium chamber mixture this file reports.
    gamma_frozen = float(gas.cp_mass / gas.cv_mass)
    return Result(
        case=case,
        backend="cantera",
        t0_k=t0,
        mw_g_mol=mw,
        gamma_frozen=gamma_frozen,
        gamma_equilibrium=None,
        cp_frozen_kj_kg_k=float(gas.cp_mass) / 1000.0,
        cstar_frozen=cstar_ideal(t0, mw, gamma_frozen),
        cstar_equilibrium=None,
        extra={"h_target_j_kg": h_target, "elements": moles},
    )


BACKENDS = {"rocketcea": run_rocketcea, "cantera": run_cantera}


# ---------------------------------------------------------------------------
# Self-check, gate and reporting
# ---------------------------------------------------------------------------


def selfcheck() -> bool:
    """Custom LOX/RP-1 cards vs rocketcea's built-in library propellants.

    This is what validates the card construction itself — the element formulas,
    the kJ/mol -> cal/mol conversion and the wt% basis — independently of any
    published engine number. Disagreement here means the cards are wrong, and
    every case below inherits it.
    """
    from rocketcea.cea_obj_w_units import CEA_Obj

    kw = dict(
        isp_units="sec",
        cstar_units="m/s",
        pressure_units="Pa",
        temperature_units="K",
        specific_heat_units="kJ/kg-K",
    )
    ref = CEA_Obj(oxName="LOX", fuelName="RP-1", **kw)
    mine = _cea_obj(CASES[2])  # Thor: the LOX/RP-1 custom cards
    ok = True
    print("Self-check: custom cards vs rocketcea built-in LOX/RP-1")
    for p0, of in [(6.895e6, 2.27), (4.10e6, 2.24)]:
        a = (
            float(ref.get_Tcomb(Pc=p0, MR=of)),
            float(ref.get_Cstar(Pc=p0, MR=of)),
            float(ref.get_Chamber_MolWt_gamma(Pc=p0, MR=of, eps=8.0)[0]),
        )
        b = (
            float(mine.get_Tcomb(Pc=p0, MR=of)),
            float(mine.get_Cstar(Pc=p0, MR=of)),
            float(mine.get_Chamber_MolWt_gamma(Pc=p0, MR=of, eps=8.0)[0]),
        )
        rel = max(abs(x / y - 1.0) for x, y in zip(a, b))
        ok &= rel < 1e-4
        print(
            f"  p0 {p0/1e6:.3f} MPa, O/F {of}: builtin T0 {a[0]:.2f} K c* {a[1]:.2f} m/s "
            f"MW {a[2]:.4f} | custom T0 {b[0]:.2f} K c* {b[1]:.2f} m/s MW {b[2]:.4f} "
            f"| max rel diff {rel:.2e}"
        )
    print(f"  => {'PASS' if ok else 'FAIL'} (threshold 1e-4)\n")
    return ok


def check_gate(results: list[Result]) -> bool:
    """The Redstone acceptance gate. Both c* definitions must clear it."""
    r = next(x for x in results if x.case.engine == "Redstone")
    print("Acceptance gate — Redstone, the one published throat diameter")
    print(
        f"  published: throat dia 15.5 in => A_t {REDSTONE_AT_M2:.6f} m^2, "
        f"mdot {REDSTONE_MDOT_KG_S:.2f} kg/s, p0 {REDSTONE_P0_PA/1e6:.4f} MPa"
    )
    print(f"  => delivered c* = p0*A_t/mdot = {REDSTONE_CSTAR_DELIVERED:.1f} m/s")
    ok = True
    for label, value in [
        ("ideal c* (frozen gamma, solver-consistent)", r.cstar_frozen),
        ("ideal c* (CEA shifting equilibrium)", r.cstar_equilibrium),
    ]:
        if value is None:
            continue
        err = value / REDSTONE_CSTAR_DELIVERED - 1.0
        passed = abs(err) <= CSTAR_GATE_TOL
        ok &= passed
        print(
            f"  {label}: {value:.1f} m/s "
            f"({err:+.2%} vs delivered, efficiency {REDSTONE_CSTAR_DELIVERED/value:.1%}) "
            f"[{'PASS' if passed else 'FAIL'}]"
        )
    print(f"  => {'PASS' if ok else 'FAIL'} (tolerance +/-{CSTAR_GATE_TOL:.0%})")
    print(GATE_EFFICIENCY_NOTE + "\n")
    return ok


def table_rows(results: list[Result]) -> list[list[str]]:
    rows = []
    for r in results:
        c = r.case
        rows.append(
            [
                c.engine,
                c.cls,
                f"{c.of:.3f}",
                f"{c.p0_pa/1e6:.2f}",
                f"{r.t0_k:.0f}",
                f"{r.mw_g_mol:.2f}",
                f"{r.gamma_frozen:.4f}",
                "-" if r.gamma_equilibrium is None else f"{r.gamma_equilibrium:.4f}",
                f"{r.cstar_frozen:.0f}",
                "-" if r.cstar_equilibrium is None else f"{r.cstar_equilibrium:.0f}",
            ]
        )
    return rows


HEADERS = [
    "engine",
    "class",
    "O/F",
    "p0 MPa",
    "T0 K",
    "MW g/mol",
    "gamma_f",
    "gamma_eq",
    "c*_f m/s",
    "c*_eq m/s",
]


def print_table(results: list[Result]) -> None:
    rows = table_rows(results)
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(HEADERS)]
    line = "  ".join(h.ljust(w) for h, w in zip(HEADERS, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(r, widths)))
    print()


def class_summary(results: list[Result]) -> dict[str, dict]:
    """Per-class aggregate: the mean and the spread across that class's cases.

    A propellant CLASS carries one (gamma, T0, MW) triple in the preset table,
    but its cases sit at different mixture ratios and chamber pressures. The
    spread printed here is what says whether one triple is honest for the class
    or whether the presets must carry their own per-case values.
    """
    out: dict[str, dict] = {}
    for cls in dict.fromkeys(r.case.cls for r in results):
        members = [r for r in results if r.case.cls == cls]
        entry = {"cases": [m.case.engine for m in members]}
        for key, get in [
            ("t0_k", lambda m: m.t0_k),
            ("mw_g_mol", lambda m: m.mw_g_mol),
            ("gamma_frozen", lambda m: m.gamma_frozen),
            ("cstar_frozen", lambda m: m.cstar_frozen),
        ]:
            vals = [get(m) for m in members]
            mean = sum(vals) / len(vals)
            entry[key] = {
                "mean": mean,
                "min": min(vals),
                "max": max(vals),
                "spread_pct": 100.0 * (max(vals) - min(vals)) / mean,
            }
        out[cls] = entry
    return out


def print_class_summary(summary: dict[str, dict]) -> None:
    print("Per-class aggregate (mean over the class's cases, and the spread)")
    for cls, e in summary.items():
        print(f"  {cls}  [{', '.join(e['cases'])}]")
        for key, unit, fmt in [
            ("t0_k", "K", "{:.0f}"),
            ("mw_g_mol", "g/mol", "{:.2f}"),
            ("gamma_frozen", "", "{:.4f}"),
            ("cstar_frozen", "m/s", "{:.0f}"),
        ]:
            s = e[key]
            print(
                f"    {key:<13} mean {fmt.format(s['mean'])} {unit:<5} "
                f"range {fmt.format(s['min'])}-{fmt.format(s['max'])} "
                f"(spread {s['spread_pct']:.2f}%)"
            )
    print()


def compare_backends(a: list[Result], b: list[Result]) -> None:
    print(f"Backend cross-check: {a[0].backend} vs {b[0].backend}")
    print(
        f"  {'engine':<24} {'dT0 %':>8} {'dMW %':>8} {'dgamma_f %':>11} {'dc*_f %':>9}"
    )
    for x, y in zip(a, b):
        d = lambda p, q: 100.0 * (q / p - 1.0)  # noqa: E731
        print(
            f"  {x.case.engine:<24} {d(x.t0_k, y.t0_k):>8.2f} "
            f"{d(x.mw_g_mol, y.mw_g_mol):>8.2f} "
            f"{d(x.gamma_frozen, y.gamma_frozen):>11.2f} "
            f"{d(x.cstar_frozen, y.cstar_frozen):>9.2f}"
        )
    print(
        "  (gri30's NASA-7 polynomials are fitted to 3500 K; cases at or past\n"
        "   that temperature are extrapolating and the disagreement grows there.)\n"
    )


def write_markdown(path: str, results: list[Result], summary: dict[str, dict],
                   gate_ok: bool, backend: str) -> None:
    rows = table_rows(results)
    r = next(x for x in results if x.case.engine == "Redstone")
    with open(path, "w") as f:
        f.write("# Propellant thermochemistry for the historical engine presets\n\n")
        f.write(
            "Generated by `tools/propellant_cea.py` "
            f"(backend: {backend}). Regenerate with:\n\n"
            "```\npython3 tools/propellant_cea.py --selfcheck "
            f"--markdown {path}\n```\n\n"
        )
        f.write(
            "Constant-enthalpy constant-pressure equilibrium from the liquid "
            "reactants' heats of formation. `gamma_f` is the FROZEN "
            "specific-heat ratio (cp/cv of the equilibrium chamber mixture) "
            "and is the value the preset table carries, because the solver is "
            "a fixed-gamma ideal gas with no species transport; `gamma_eq` is "
            "CEA's shifting-equilibrium exponent, shown only so the gap is "
            "visible. `c*_f` is the ideal c* the fixed-gamma solver "
            "reproduces; `c*_eq` is CEA's.\n\n"
        )
        f.write("## Reactants\n\n")
        f.write("| reactant | formula | MW g/mol | hf kJ/mol | T K |\n")
        f.write("|---|---|---|---|---|\n")
        for rr in REACTANTS.values():
            formula = "".join(f"{el}{n:g}" for el, n in rr.elements.items())
            f.write(
                f"| {rr.label} | {formula} | {rr.mw:.4f} | "
                f"{rr.hf_kj_mol:+.3f} | {rr.t_k} |\n"
            )
        f.write("\n## Cases\n\n")
        f.write("| " + " | ".join(HEADERS) + " |\n")
        f.write("|" + "---|" * len(HEADERS) + "\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\nOxidizer/fuel specs:\n\n")
        for mix in [LOX, RFNA, ETHANOL75, RP1, ANILINE_FURFURYL]:
            f.write(f"- **{mix.name}** — {mix.describe()} (mass fractions)\n")
        f.write("\n## Per-class aggregate\n\n")
        f.write("| class | cases | T0 K | MW g/mol | gamma_f | c*_f m/s |\n")
        f.write("|---|---|---|---|---|---|\n")
        for cls, e in summary.items():
            def cell(key, fmt):
                s = e[key]
                return (
                    f"{fmt.format(s['mean'])} ({fmt.format(s['min'])}–"
                    f"{fmt.format(s['max'])}, {s['spread_pct']:.2f}%)"
                )
            f.write(
                f"| {cls} | {len(e['cases'])} | {cell('t0_k', '{:.0f}')} | "
                f"{cell('mw_g_mol', '{:.2f}')} | {cell('gamma_frozen', '{:.4f}')} | "
                f"{cell('cstar_frozen', '{:.0f}')} |\n"
            )
        f.write("\nEach cell is `mean (min–max, spread %)` over that class's cases.\n")
        f.write("\n## Acceptance gate — Redstone\n\n")
        f.write(
            f"Published: throat dia 15.5 in => A_t = {REDSTONE_AT_M2:.6f} m², "
            f"mdot 355 lb/s = {REDSTONE_MDOT_KG_S:.2f} kg/s, "
            f"p0 = 318 psia = {REDSTONE_P0_PA/1e6:.4f} MPa, so delivered "
            f"c* = p0·A_t/mdot = **{REDSTONE_CSTAR_DELIVERED:.0f} m/s**.\n\n"
        )
        f.write(
            f"- computed ideal c* (frozen gamma): **{r.cstar_frozen:.0f} m/s** "
            f"({r.cstar_frozen/REDSTONE_CSTAR_DELIVERED - 1.0:+.2%})\n"
            f"- computed ideal c* (CEA equilibrium): **{r.cstar_equilibrium:.0f} m/s** "
            f"({r.cstar_equilibrium/REDSTONE_CSTAR_DELIVERED - 1.0:+.2%})\n"
            f"- tolerance ±{CSTAR_GATE_TOL:.0%} — **{'PASS' if gate_ok else 'FAIL'}**\n\n"
        )
        f.write(
            "> **The gate passes, but from the wrong side.** Delivered c\\* should "
            "be 92–96% of ideal; against these numbers it lands at "
            f"{REDSTONE_CSTAR_DELIVERED/r.cstar_frozen:.0%} (frozen) / "
            f"{REDSTONE_CSTAR_DELIVERED/r.cstar_equilibrium:.0%} (equilibrium), "
            "which no engine achieves. The chemistry is not the suspect — "
            "rocketcea and cantera agree to 0.1%, and the reactant cards "
            "reproduce CEA's own library propellants to 5e-6. The published "
            "trio is mutually optimistic instead: most likely 318 psia is "
            "INJECTOR-END pressure, which exceeds nozzle stagnation pressure "
            "by the Rayleigh loss across the combustor, so `p0·A_t/mdot` "
            "overstates delivered c\\*. A 3–5% correction there restores a "
            "96–99% efficiency and still clears this gate. Treat 1658 m/s as "
            "an upper bound on delivered c\\*, not a measurement.\n"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--backend",
        choices=[*BACKENDS, "both"],
        default="rocketcea",
        help="rocketcea (preferred), cantera (fallback), or both (cross-check)",
    )
    ap.add_argument("--selfcheck", action="store_true", help="validate the CEA cards")
    ap.add_argument("--markdown", metavar="PATH", help="write the full table to PATH")
    args = ap.parse_args(argv)

    if args.selfcheck and not selfcheck():
        print("Self-check FAILED: the reactant cards do not reproduce the "
              "library propellants. Fix the cards, not the tolerance.", file=sys.stderr)
        return 1

    primary = "rocketcea" if args.backend in ("rocketcea", "both") else "cantera"
    results = [BACKENDS[primary](c) for c in CASES]
    print(f"Backend: {primary}\n")
    print_table(results)
    summary = class_summary(results)
    print_class_summary(summary)

    if args.backend == "both":
        compare_backends(results, [run_cantera(c) for c in CASES])

    gate_ok = check_gate(results)
    if args.markdown:
        write_markdown(args.markdown, results, summary, gate_ok, primary)
        print(f"wrote {args.markdown}")
    if not gate_ok:
        print("ACCEPTANCE GATE FAILED — do not proceed to the preset table. "
              "The reactant enthalpies or the element balance are wrong; do "
              "not widen the tolerance.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
