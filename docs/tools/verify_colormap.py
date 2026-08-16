#!/usr/bin/env python3
"""Verify a colormap: perceptual uniformity, lightness monotonicity, CVD,
grayscale survival, and the minimal anchor set needed to reproduce it.

    pip install colorspacious cmcrameri --break-system-packages

    # named map from matplotlib / cmcrameri
    python verify_colormap.py viridis
    python verify_colormap.py cmc.batlow

    # a hand-authored anchor table (pos,r,g,b per line, rgb 0-255)
    python verify_colormap.py --anchors my_map.csv

    # generate a Moreland Msh diverging map between two endpoints
    python verify_colormap.py --diverging 59,76,192 180,4,38

Add --plot out.png for swatch + CVD + L* + step plots.
"""
import argparse, sys
import numpy as np
from colorspacious import cspace_convert, deltaE

N = 256

# ------------------------------------------------------- Moreland Msh (paper)
def rgb2msh(rgb):
    L, a, b = cspace_convert(np.asarray(rgb, float), "sRGB1", "CIELab")
    M = float(np.sqrt(L * L + a * a + b * b))
    s = float(np.arccos(L / M)) if M > 1e-12 else 0.0
    return np.array([M, s, float(np.arctan2(b, a))])

def msh2rgb(msh):
    M, s, h = msh
    lab = np.array([M * np.cos(s), M * np.sin(s) * np.cos(h), M * np.sin(s) * np.sin(h)])
    return np.clip(cspace_convert(lab, "CIELab", "sRGB1"), 0, 1)

def _rad_diff(a, b):
    d = abs(a - b)
    return d if d <= np.pi else 2 * np.pi - d

def _adjust_hue(msh_sat, M_unsat):
    M, s, h = msh_sat
    if M >= M_unsat:
        return h
    spin = s * np.sqrt(max(M_unsat ** 2 - M * M, 0.0)) / (M * np.sin(s)) if s > 1e-12 else 0.0
    return h + spin if h > -np.pi / 3 else h - spin

def moreland_diverging(rgb_lo, rgb_hi, n=N):
    """Moreland 2009, Fig. 13 InterpolateColor."""
    out = np.zeros((n, 3))
    for i, x in enumerate(np.linspace(0, 1, n)):
        m1, m2, t = rgb2msh(rgb_lo).copy(), rgb2msh(rgb_hi).copy(), x
        if m1[1] > 0.05 and m2[1] > 0.05 and _rad_diff(m1[2], m2[2]) > np.pi / 3:
            Mmid = max(m1[0], m2[0], 88.0)
            if t < 0.5:
                m2, t = np.array([Mmid, 0.0, 0.0]), 2 * t
            else:
                m1, t = np.array([Mmid, 0.0, 0.0]), 2 * t - 1
        if m1[1] < 0.05 and m2[1] > 0.05:
            m1[2] = _adjust_hue(m2, m1[0])
        elif m2[1] < 0.05 and m1[1] > 0.05:
            m2[2] = _adjust_hue(m1, m2[0])
        out[i] = msh2rgb((1 - t) * m1 + t * m2)
    return out

# ------------------------------------------------------------------- loading
def load(name, n=N):
    xs = np.linspace(0, 1, n)
    if name.startswith("cmc."):
        import cmcrameri.cm as cmc
        f = getattr(cmc, name[4:])
    else:
        import matplotlib.pyplot as plt
        f = plt.get_cmap(name)
    return np.array([f(x)[:3] for x in xs])

def load_anchors(path, n=N):
    pts = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        p, r, g, b = [float(v) for v in line.replace(",", " ").split()]
        pts.append((p, np.array([r, g, b]) / 255.0))
    pos = np.array([p for p, _ in pts])
    c = np.array([v for _, v in pts])
    xs = np.linspace(0, 1, n)
    return np.clip(np.stack([np.interp(xs, pos, c[:, k]) for k in range(3)], 1), 0, 1)

# ------------------------------------------------------------------- metrics
def lab_of(rgb):
    return cspace_convert(rgb, "sRGB1", "CIELab")

def cvd(rgb, kind, sev=100):
    return np.clip(cspace_convert(
        rgb, {"name": "sRGB1+CVD", "cvd_type": kind, "severity": sev}, "sRGB1"), 0, 1)

def dE(a, b):
    return deltaE(a, b, input_space="sRGB1", uniform_space="CAM02-UCS")

def steps(rgb):
    return np.array([dE(rgb[i], rgb[i + 1]) for i in range(len(rgb) - 1)])

def metrics(rgb):
    L = lab_of(rgb)[:, 0]
    dL = np.diff(L)
    st = steps(rgb)
    m = {
        "L*_start": round(float(L[0]), 1), "L*_end": round(float(L[-1]), 1),
        "L*_monotone": bool(np.all(dL >= -0.15) or np.all(dL <= 0.15)),
        "L*_reversals": int(np.sum(np.sign(dL[:-1]) * np.sign(dL[1:]) < 0)),
        "dE00_total": round(float(st.sum()), 1),
        "dE00_step_cv_pct": round(float(100 * st.std() / st.mean()), 1),
        "dE00_peak_over_mean": round(float(st.max() / st.mean()), 2),
        "worst_step_at_pos": round(float(int(np.argmax(st)) / (len(rgb) - 1)), 3),
    }
    for kind, tag in [("deuteranomaly", "deut"), ("protanomaly", "prot")]:
        sc = steps(cvd(rgb, kind))
        m[f"{tag}_dE00_total"] = round(float(sc.sum()), 1)
        m[f"{tag}_min_step"] = round(float(sc.min()), 3)
    grayrgb = np.clip(cspace_convert(
        np.stack([L, np.zeros(len(rgb)), np.zeros(len(rgb))], 1), "CIELab", "sRGB1"), 0, 1)
    m["grayscale_dE00_total"] = round(float(steps(grayrgb).sum()), 1)
    return m

# -------------------------------------------------------- minimal anchor fit
def _reconstruct(idx, ref):
    n = len(ref)
    xs = np.arange(n) / (n - 1)
    c = np.round(ref[idx] * 255.0) / 255.0   # anchors quantised to u8, as shipped
    lerp = np.stack([np.interp(xs, xs[idx], c[:, k]) for k in range(3)], 1)
    # build() writes a u8 LUT, so every ENTRY is rounded, not just the anchors.
    # Interpolating in float here reports an error smaller than what actually
    # ships -- it read 1.96 for inferno where the shipped LUT measures 2.12.
    return np.clip(np.round(lerp * 255.0) / 255.0, 0, 1)

def fit_anchors(ref, tol=2.0, cap=24):
    idx = [0, len(ref) - 1]
    while True:
        err = dE(ref, _reconstruct(np.array(idx), ref))
        if err.max() <= tol or len(idx) >= cap:
            return sorted(idx), float(err.max())
        j = int(np.argmax(err))
        if j in idx:
            return sorted(idx), float(err.max())
        idx = sorted(idx + [j])

# ---------------------------------------------------------------------- main
def verdict(m, kind):
    out = []
    if kind == "sequential":
        if not m["L*_monotone"]:
            out.append(f"FAIL  lightness not monotonic ({m['L*_reversals']} reversals)")
    elif m["L*_reversals"] > 1:
        out.append(f"FAIL  {m['L*_reversals']} lightness reversals; a diverging map should have 1")
    if m["dE00_step_cv_pct"] > 15:
        out.append(f"WARN  step CV {m['dE00_step_cv_pct']}% (>15%): uneven visual resolution")
    if m["dE00_peak_over_mean"] > 2:
        out.append(f"WARN  peak/mean step {m['dE00_peak_over_mean']} at pos {m['worst_step_at_pos']}: likely Mach band")
    if m["deut_min_step"] < 0.1:
        out.append(f"WARN  deuteranopia min step {m['deut_min_step']}: adjacent values merge for ~5% of men")
    return out or ["PASS"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", help="matplotlib name, or cmc.<name> for Crameri")
    ap.add_argument("--anchors", help="CSV of pos,r,g,b (rgb 0-255)")
    ap.add_argument("--diverging", nargs=2, metavar=("LO", "HI"), help="r,g,b r,g,b -> Moreland Msh map")
    ap.add_argument("--kind", choices=["sequential", "diverging"], default="sequential")
    ap.add_argument("--tol", type=float, default=2.0, help="max dE00 for the anchor fit")
    ap.add_argument("--plot", help="write a verification PNG here")
    a = ap.parse_args()

    if a.diverging:
        p = [np.array([float(v) for v in s.split(",")]) / 255.0 for s in a.diverging]
        rgb, label, a.kind = moreland_diverging(p[0], p[1]), "moreland-diverging", "diverging"
    elif a.anchors:
        rgb, label = load_anchors(a.anchors), a.anchors
    elif a.name:
        rgb, label = load(a.name), a.name
    else:
        ap.error("give a name, --anchors, or --diverging")

    m = metrics(rgb)
    idx, err = fit_anchors(rgb, tol=a.tol)

    print(f"\n{label}  ({a.kind})\n" + "-" * 60)
    for k, v in m.items():
        print(f"  {k:<24} {v}")
    print()
    for line in verdict(m, a.kind):
        print("  " + line)
    print(f"\n  minimal anchors: {len(idx)}  (max dE00 {err:.2f} vs the full table)\n")
    print("| pos | hex | R, G, B |")
    print("|---|---|---|")
    for i in idx:
        r, g, b = [int(round(v * 255)) for v in rgb[i]]
        print(f"| {i/(len(rgb)-1):.4f} | `#{r:02X}{g:02X}{b:02X}` | {r}, {g}, {b} |")

    if a.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 1, figsize=(8, 5), height_ratios=[1, 1, 2, 2])
        ax[0].imshow(rgb[None], aspect="auto"); ax[0].set_ylabel("sRGB", rotation=0, ha="right")
        ax[1].imshow(cvd(rgb, "deuteranomaly")[None], aspect="auto")
        ax[1].set_ylabel("deut", rotation=0, ha="right")
        ax[2].plot(lab_of(rgb)[:, 0], "k"); ax[2].set_ylabel("L*"); ax[2].set_ylim(0, 100)
        ax[3].plot(steps(rgb), "crimson"); ax[3].set_ylabel("dE00 step"); ax[3].set_ylim(bottom=0)
        for x in ax[:2]:
            x.set_xticks([]); x.set_yticks([])
        fig.suptitle(label)
        fig.savefig(a.plot, dpi=120, bbox_inches="tight")
        print(f"\n  wrote {a.plot}")

if __name__ == "__main__":
    sys.exit(main())
