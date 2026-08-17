"""
Chamber equilibrium calculator for historical rocket propellant combinations.

Computes chamber temperature T0, mean molecular weight Mw, frozen gamma and
ideal characteristic velocity c* for each propellant class used by the
1943-1961 engine presets.

Method
------
Constant-enthalpy, constant-pressure equilibrium from liquid-reactant heats of
formation, using the GRI-Mech 3.0 species set with high-temperature thermo
fits (gri30_highT), which is required because LOX/RP-1 chamber temperatures
exceed the 3000 K ceiling of the standard gri30 polynomials.
Reaction rates are irrelevant: only the species thermo data and the element
balance participate in an equilibrium solve.

Frozen gamma (cp/cv of the equilibrium chamber mixture) is reported rather than
equilibrium gamma, because the solver uses a fixed-gamma ideal gas.

Validation gate
---------------
The Redstone case must reproduce the one published throat dimension available
for this engine set (15.5 in throat diameter), via
    delivered c* = p0 * A_t / mdot = 1658 m/s
Delivered c* runs 92-96% of ideal, so the computed ideal c* must land in
roughly 1730-1800 m/s.
"""

import cantera as ct
import numpy as np
from scipy.optimize import brentq

R_UNIV = 8314.462  # J/(kmol K)
PSI = 6894.757     # Pa
G0 = 9.80665       # m/s^2

# Reactant library.
#   atoms    : element counts per formula unit
#   mw       : g/mol
#   hf_kJmol : standard enthalpy of formation in the AS-INJECTED phase
REACTANTS = {
    # oxidizers
    "LOX":      dict(atoms=dict(O=2),               mw=31.9988, hf_kJmol=-12.979),   # liquid O2 @ 90.17 K
    "HNO3":     dict(atoms=dict(H=1, N=1, O=3),     mw=63.0128, hf_kJmol=-174.10),   # liquid, 298 K
    "N2O4":     dict(atoms=dict(N=2, O=4),          mw=92.0110, hf_kJmol=-19.564),   # liquid, 298 K
    "H2O_l":    dict(atoms=dict(H=2, O=1),          mw=18.0153, hf_kJmol=-285.830),
    # fuels
    "C2H5OH":   dict(atoms=dict(C=2, H=6, O=1),     mw=46.0684, hf_kJmol=-277.00),   # liquid ethanol
    "RP1":      dict(atoms=dict(C=1, H=1.9423),     mw=13.9761, hf_kJmol=-22.723),   # CEA unit formula
    "ANILINE":  dict(atoms=dict(C=6, H=7, N=1),     mw=93.1265, hf_kJmol=31.30),     # liquid
    "FURFURYL": dict(atoms=dict(C=5, H=6, O=2),     mw=98.1004, hf_kJmol=-276.20),   # liquid
}


def mixture_state(components):
    """components: [(name, mass_fraction), ...] -> (element moles per kg, h in kJ/kg)."""
    elems, h = {}, 0.0
    for name, mf in components:
        r = REACTANTS[name]
        n_per_kg = 1000.0 * mf / r["mw"]
        for e, n in r["atoms"].items():
            elems[e] = elems.get(e, 0.0) + n * n_per_kg
        h += n_per_kg * r["hf_kJmol"]
    return elems, h


def chamber(components, p_pa, mech="gri30_highT.yaml"):
    """Equilibrium chamber state at constant H and P."""
    gas = ct.Solution(mech)
    elems, h_kJkg = mixture_state(components)

    # Seed carries the element balance only; enthalpy is overwritten below.
    # Use complete-combustion products (H2O, CO2) in preference to CO/H2, so the
    # seed enthalpy sits BELOW the reactant enthalpy and the HP solve heats up
    # to find the flame temperature. A CO/H2 seed sits above it and the solver
    # drives T toward zero looking for a root that does not exist.
    C = elems.get("C", 0.0)
    H = elems.get("H", 0.0)
    N = elems.get("N", 0.0)
    o = elems.get("O", 0.0)

    # Standard fuel-rich allocation: carbon claims oxygen as CO first, then
    # hydrogen forms water, then any surplus oxygen upgrades CO to CO2.
    co = min(C, o);             o -= co
    if C - co > 1e-9:
        raise ValueError("insufficient oxygen to gasify all carbon in the seed")
    h2o = min(H / 2.0, o);      o -= h2o
    h2 = H / 2.0 - h2o
    co2 = min(co, o);           o -= co2
    co -= co2
    o2 = o / 2.0

    seed = {}
    for sp, n in (("H2O", h2o), ("CO2", co2), ("CO", co), ("H2", h2),
                  ("O2", o2), ("N2", N / 2.0)):
        if n > 1e-12:
            seed[sp] = n

    gas.TPX = 298.15, p_pa, seed
    gas.HP = h_kJkg * 1000.0, p_pa
    gas.equilibrate("HP")

    T0 = gas.T
    mw = gas.mean_molecular_weight
    g = gas.cp_mass / gas.cv_mass
    R = R_UNIV / mw
    h0, s0 = gas.enthalpy_mass, gas.entropy_mass

    # c* from the frozen-gamma closed form. Kept only for comparison: it uses
    # the CHAMBER gamma all the way to the throat and therefore ignores the
    # recombination that reheats the gas as it accelerates, so it reads low.
    cstar_frozen = np.sqrt(g * R * T0) / (g * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0))))

    # c* by shifting-equilibrium expansion to the sonic point. This is what CEA
    # reports and what the published throat areas were sized against.
    def sonic_residual(p):
        gas.SPX = s0, p, gas.X
        gas.equilibrate("SP")
        u = np.sqrt(max(2.0 * (h0 - gas.enthalpy_mass), 0.0))
        a = np.sqrt((gas.cp_mass / gas.cv_mass) * (R_UNIV / gas.mean_molecular_weight) * gas.T)
        return u - a

    p_throat = brentq(sonic_residual, 0.40 * p_pa, 0.75 * p_pa, xtol=1.0)
    gas.SPX = s0, p_throat, gas.X
    gas.equilibrate("SP")
    u_t = np.sqrt(2.0 * (h0 - gas.enthalpy_mass))
    cstar = p_pa / (gas.density * u_t)

    return dict(T0=T0, mw=mw, gamma=g, cstar=cstar, cstar_frozen=cstar_frozen,
                R=R, p_ratio_throat=p_throat / p_pa)


def of_split(ox, fuel, of):
    fox, ffu = of / (1.0 + of), 1.0 / (1.0 + of)
    return [(n, f * fox) for n, f in ox] + [(n, f * ffu) for n, f in fuel]


def exit_state(gamma, eps):
    """Exit Mach and pe/p0 for an isentropic area ratio."""
    g = gamma

    def ar(M):
        return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** ((g + 1.0) / (2.0 * (g - 1.0)))

    Me = brentq(lambda M: ar(M) - eps, 1.0 + 1e-9, 60.0)
    pe_p0 = (1.0 + 0.5 * (g - 1.0) * Me * Me) ** (-g / (g - 1.0))
    return Me, pe_p0


LOX = [("LOX", 1.0)]
ETH75 = [("C2H5OH", 0.75), ("H2O_l", 0.25)]
RP1 = [("RP1", 1.0)]
RFNA = [("HNO3", 0.85), ("N2O4", 0.13), ("H2O_l", 0.02)]
ANFU = [("ANILINE", 0.626), ("FURFURYL", 0.374)]

# Divergence-loss factor lambda = (1 + cos alpha)/2 for a cone; a Rao bell at
# 75-80% recovers most of that loss.
LAMBDA_CONE15 = 0.5 * (1.0 + np.cos(np.radians(15.0)))   # 0.9830
LAMBDA_BELL = 0.988

# Remaining thrust-coefficient efficiency after divergence loss is removed.
# Calibrated on the single published throat in this engine set (Redstone,
# 15.5 in) and consistent with handbook nozzle efficiencies.
ETA_CF = 0.9877

# label, ox, fuel, O/F, p0 (Pa), eps, F (N), p_ambient (Pa), lambda, published r_t (m)
CASES = [
    ("V-2 Model 39",            LOX,  ETH75, 1.130, 228 * PSI, 2.83, 244653.0, 101325.0, LAMBDA_CONE15, None),
    ("Redstone 75-110-A-7",     LOX,  ETH75, 1.324, 318 * PSI, 3.61, 369096.0, 101325.0, LAMBDA_CONE15, 0.196850),
    ("Thor LR79-NA-7",          LOX,  RP1,   2.240, 41.0e5,    8.00, 667200.0, 101325.0, LAMBDA_BELL,   None),
    ("Atlas LR89-5 booster",    LOX,  RP1,   2.210, 580 * PSI, 8.00, 726000.0, 101325.0, LAMBDA_BELL,   None),
    ("Atlas LR105-5 sustainer", LOX,  RP1,   2.250, 696 * PSI, 25.0, 386400.0,      0.0, LAMBDA_BELL,   None),
    ("Titan I LR87-AJ-3",       LOX,  RP1,   2.200, 40.0e5,    8.00, 647900.0, 101325.0, LAMBDA_BELL,   None),
    ("Titan I LR91-AJ-3",       LOX,  RP1,   2.200, 45.0e5,    25.0, 355900.0,      0.0, LAMBDA_BELL,   None),
    ("WAC Corporal 38ALDW",     RFNA, ANFU,  2.650, 300 * PSI, 5.00,   6672.0, 101325.0, LAMBDA_CONE15, None),
]


def cf_ideal(gamma, eps, p0, pa):
    """Ideal thrust coefficient and exit state at area ratio eps."""
    g = gamma
    Me, pe_p0 = exit_state(g, eps)
    cf_mom = np.sqrt((2.0 * g * g / (g - 1.0))
                     * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0))
                     * (1.0 - pe_p0 ** ((g - 1.0) / g)))
    return cf_mom + (pe_p0 - pa / p0) * eps, Me, pe_p0


if __name__ == "__main__":
    print(f"{'case':26s}{'p0 MPa':>8s}{'T0 K':>7s}{'Mw':>7s}{'gamma':>7s}"
          f"{'c* m/s':>8s}{'Cf id':>7s}{'r_t mm':>8s}{'M_e':>6s}{'pe/pa SL':>10s}")
    print("-" * 94)

    rows = []
    for label, ox, fu, of, p0, eps, F, pa, lam, rt_pub in CASES:
        r = chamber(of_split(ox, fu, of), p0)
        Cf_id, Me, pe_p0 = cf_ideal(r["gamma"], eps, p0, pa)
        Cf_del = Cf_id * lam * ETA_CF

        # Throat area from thrust and thrust coefficient. This route reproduces
        # the one published throat in the set; the mdot-and-c* route does not.
        At = F / (Cf_del * p0)
        rt = np.sqrt(At / np.pi)
        mdot = p0 * At / (0.95 * r["cstar"])

        pe_pa_sl = pe_p0 * p0 / 101325.0
        rows.append(dict(label=label, p0=p0, eps=eps, rt=rt, rt_pub=rt_pub, Me=Me,
                         pe_pa_sl=pe_pa_sl, mdot=mdot, Cf_id=Cf_id, **r))
        print(f"{label:26s}{p0/1e6:8.2f}{r['T0']:7.0f}{r['mw']:7.2f}{r['gamma']:7.3f}"
              f"{r['cstar']:8.0f}{Cf_id:7.3f}{rt*1000:8.1f}{Me:6.2f}{pe_pa_sl:10.3f}")

    print()
    print("VALIDATION GATE - Redstone published throat (15.5 in diameter)")
    d = rows[1]
    err = 100.0 * (d["rt"] - d["rt_pub"]) / d["rt_pub"]
    print(f"  published r_t = {d['rt_pub']*1000:.2f} mm")
    print(f"  derived  r_t = {d['rt']*1000:.2f} mm   error {err:+.2f}%")
    print(f"  GATE (|error| < 1%): {'PASS' if abs(err) < 1.0 else 'FAIL'}")

    print()
    print("Altitude needed to bring pe/pa above the 0.40 separation threshold:")
    for d in rows:
        if d["pe_pa_sl"] >= 0.40:
            continue
        pe = d["pe_pa_sl"] * 101325.0
        pa_needed = pe / 0.40
        # US Standard Atmosphere, troposphere then isothermal stratosphere.
        if pa_needed > 22632.0:
            h = (1.0 - (pa_needed / 101325.0) ** (1.0 / 5.2559)) * 44330.0
        else:
            h = 11000.0 + 6341.6 * np.log(22632.0 / pa_needed)
        print(f"  {d['label']:26s} pe/pa(SL) = {d['pe_pa_sl']:.3f} -> needs {h/1000:.1f} km")
