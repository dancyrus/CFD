"""Recompute the acceptance-ladder G1 and G2 references from theta-beta-M,
Rankine-Hugoniot and Prandtl-Meyer. Nothing here is read off a chart or copied
from a table, which is the point: the rungs in cfd-core/tests/ladder.rs assert
these digits, so the digits have to have a derivation that can be re-run.

    python3 docs/work-orders/general-geometry-refs.py

Standard library only (no scipy) so it runs anywhere the repo does. The
bisection below is deliberately dumb and deliberately over-iterated: these are
one-off constants, not a hot loop."""
from math import sin, cos, tan, atan, asin, sqrt, radians, degrees, pi

def brentq(f, a, b, xtol=1e-15, rtol=1e-15):
    fa, fb = f(a), f(b)
    assert fa * fb <= 0, (a, b, fa, fb)
    for _ in range(400):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)

G = 1.4

def theta_from_beta(beta, M, g=G):
    """theta-beta-M, beta in radians."""
    return atan(2.0 / tan(beta) * (M * M * sin(beta) ** 2 - 1.0)
                / (M * M * (g + cos(2 * beta)) + 2.0))

def beta_weak(theta, M, g=G):
    mu = asin(1.0 / M)                      # Mach angle: theta = 0 branch start
    # weak solution lies between mu and the detachment beta
    lo, hi = mu + 1e-12, radians(89.999999)
    # find beta_max (theta maximum) by golden search, then bracket on [mu, beta_max]
    a, b = lo, hi
    gr = (sqrt(5) - 1) / 2
    for _ in range(300):
        c, d = b - gr * (b - a), a + gr * (b - a)
        if theta_from_beta(c, M, g) < theta_from_beta(d, M, g):
            a = c
        else:
            b = d
    beta_max = (a + b) / 2
    return brentq(lambda x: theta_from_beta(x, M, g) - theta, lo, beta_max, xtol=1e-15, rtol=1e-15)

def oblique(M1, theta, g=G):
    """Returns beta[deg], p2/p1, rho2/rho1, M2."""
    beta = beta_weak(theta, M1, g)
    Mn1 = M1 * sin(beta)
    pr = 1.0 + 2.0 * g / (g + 1.0) * (Mn1 ** 2 - 1.0)
    rr = (g + 1.0) * Mn1 ** 2 / ((g - 1.0) * Mn1 ** 2 + 2.0)
    Mn2 = sqrt((1.0 + 0.5 * (g - 1.0) * Mn1 ** 2) / (g * Mn1 ** 2 - 0.5 * (g - 1.0)))
    M2 = Mn2 / sin(beta - theta)
    return degrees(beta), pr, rr, M2

print("=== G1: symmetric double wedge, 10 deg half-angle each, M_inf = 2.0 ===")
th = radians(10.0)
b1, p21, r21, M2 = oblique(2.0, th)
print(f"incident  beta_1 = {b1:.6f} deg   p2/p1 = {p21:.6f}   M2 = {M2:.6f}")
b2, p32, r32, M3 = oblique(M2, th)
print(f"reflected beta_2 = {b2:.6f} deg (to the LOCAL flow)   p3/p2 = {p32:.6f}   M3 = {M3:.6f}")
print(f"p3/p1 = {p21 * p32:.6f}")
print(f"reflected shock angle to the AXIS = {b2 - degrees(th):.6f} deg")
print(f"tan(beta_1) = {tan(radians(b1)):.9f}")

print()
print("=== G2: symmetric diamond airfoil, 5 deg half-angle, M_inf = 2.0 ===")
eps = radians(5.0)
tc = tan(eps)
print(f"t/c = {tc:.6f}")
bLE, p2, _, M2d = oblique(2.0, eps)
print(f"LE shock beta = {bLE:.6f} deg,  fore p/p_inf = {p2:.6f},  M2 = {M2d:.6f}")

def nu(M, g=G):
    k = sqrt((g + 1.0) / (g - 1.0))
    return k * atan(sqrt((M * M - 1.0) / k / k)) - atan(sqrt(M * M - 1.0))

def M_from_nu(v, g=G):
    return brentq(lambda M: nu(M, g) - v, 1.0 + 1e-12, 60.0, xtol=1e-14, rtol=1e-15)

# Expansion through 2*eps at the shoulder.
M3d = M_from_nu(nu(M2d) + 2.0 * eps)
p0_p = lambda M, g=G: (1.0 + 0.5 * (g - 1.0) * M * M) ** (g / (g - 1.0))
p3 = p2 * p0_p(M2d) / p0_p(M3d)
print(f"shoulder expansion {degrees(2*eps):.1f} deg -> M3 = {M3d:.6f},  aft p/p_inf = {p3:.6f}")

q = 0.5 * G * 2.0 ** 2          # q_inf / p_inf
cd = (p2 - p3) / q * tc
print(f"q_inf/p_inf = {q}")
print(f"exact shock-expansion C_d = {cd:.6f}")
print(f"Ackeret linearised C_d = {4.0 * tc ** 2 / sqrt(2.0 ** 2 - 1.0):.6f}")

print()
print("=== G0 sanity: M = 0.3, no shocks -> d'Alembert C_d = 0 exactly ===")
print("=== G3: Woodward & Colella, M = 3, gamma = 1.4 ===")
print(f"u_inf (rho=1, p=1 nondim) = {3.0 * sqrt(G):.9f}, a_inf = {sqrt(G):.9f}")
