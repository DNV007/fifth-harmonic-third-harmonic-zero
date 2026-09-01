"""Second physical realization: an RLC network with inductive (mutual-inductance)
coupling exhibits the same exact third-harmonic transmission zero.

Model (charge variables q1,q2; two LC resonators, mutual inductance LAMBDA, on-site
nonlinear capacitor b1, nonlinear coupling capacitor GC):

  q1'' + 2g1 q1' + w1^2 q1 + b1 q1^3 + LAMBDA q2'' + GC (q1-q2)^3 = E0 cos(Om t)
  q2'' + 2g2 q2'  + w2^2 q2         + LAMBDA q1'' + GC (q2-q1)^3 = 0

The coupling enters at q'' (off-diagonal of the dynamic-stiffness matrix
~ -LAMBDA (kOm)^2), a DIFFERENT linear network from the mechanical stiffness coupling.
The projected-cancellation criterion Y3 = c^dag D3^-1 (S_on-site + S_coupling) = 0
still has an exact solution at finite forcing -- demonstrating the principle in a
second realization. Outputs the working point, the full-rank drive Jacobian, and
two figures: an Argand trajectory of X3 through the origin and a two-parameter
zero map.

Run:  uv run python scripts/run_k3_rlc_second_example.py
"""
import os
import numpy as np
from scipy.optimize import least_squares, root
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W1, W2 = 1.0, 1.40
G1, G2 = 0.020, 0.025
LAMBDA, GC = 0.15, -0.25   # LAMBDA = mutual-inductance coupling (SM Sec. S9)
N, NT = 7, 256  # reliable; the finite-amplitude zero is bistable, needs stable projection
_D = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(_D, "..", "figures")

nn = np.arange(NT)
PH = 2 * np.pi * np.outer(np.arange(1, N + 1), nn) / NT
COS, SIN = np.cos(PH), np.sin(PH)
MVEC = np.arange(1, N + 1)[:, None]
IC, IS = 2 * (3 - 1), 2 * (3 - 1) + 1


def recon(c, Om):
    x1 = c[:2 * N].reshape(N, 2); x2 = c[2 * N:].reshape(N, 2)
    a1, b1 = x1[:, [0]], x1[:, [1]]; a2, b2 = x2[:, [0]], x2[:, [1]]
    q1 = np.sum(a1 * COS + b1 * SIN, 0); q2 = np.sum(a2 * COS + b2 * SIN, 0)
    v1 = np.sum(MVEC * Om * (-a1 * SIN + b1 * COS), 0); v2 = np.sum(MVEC * Om * (-a2 * SIN + b2 * COS), 0)
    ac1 = np.sum(-(MVEC * Om) ** 2 * (a1 * COS + b1 * SIN), 0); ac2 = np.sum(-(MVEC * Om) ** 2 * (a2 * COS + b2 * SIN), 0)
    return q1, v1, ac1, q2, v2, ac2


def hb_res(c, Om, b1, E0, gc=GC):
    q1, v1, a1, q2, v2, a2 = recon(c, Om)
    r1 = a1 + 2 * G1 * v1 + W1 ** 2 * q1 + b1 * q1 ** 3 + LAMBDA * a2 + gc * (q1 - q2) ** 3 - E0 * COS[0]
    r2 = a2 + 2 * G2 * v2 + W2 ** 2 * q2 + LAMBDA * a1 + gc * (q2 - q1) ** 3
    out = np.empty(4 * N)
    for m in range(N):
        out[4 * m:4 * m + 4] = [(2 / NT) * (r1 @ COS[m]), (2 / NT) * (r1 @ SIN[m]),
                                (2 / NT) * (r2 @ COS[m]), (2 / NT) * (r2 @ SIN[m])]
    return out


def solve_hb(Om, b1, E0, guess=None, gc=GC):
    # warm-started; a good guess is essential -- the finite-amplitude zero is bistable.
    z0 = np.zeros(4 * N) if guess is None else guess.copy()
    if guess is None:
        z0[0] = E0 / (W1 ** 2 - Om ** 2)
    if guess is not None:  # warm start -> local Newton stays on the orbit
        sol = root(lambda c: hb_res(c, Om, b1, E0, gc), z0, method="hybr", tol=1e-13)
        if sol.success:
            return sol.x
    return least_squares(lambda c: hb_res(c, Om, b1, E0, gc), z0, xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=4000).x


def Y3(c):
    return complex(c[IC], -c[IS])


def solve_zero(E0, seed):
    """Free (Om,b1) so that node-1 third harmonic vanishes."""
    def aug(z):
        c = z[:4 * N]
        return np.concatenate([hb_res(c, float(z[-2]), float(z[-1]), E0), [c[IC], c[IS]]])
    s = least_squares(aug, seed, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=9000)
    return s.x


def main():
    # acquire the working point by warm-stepping E0 from low forcing along the
    # clean branch (grid 0.5->2.2 reaches the modest-b1 zero; other grids jump to
    # notches or far large-b1 roots -- the finite-amplitude zero is bistable).
    E0 = 2.0
    z = np.concatenate([solve_hb(0.291, 0.455, 0.9), [0.291, 0.455]])
    zc = None
    for E0s in np.linspace(0.5, 2.2, 35):
        z = solve_zero(E0s, z)
        if abs(E0s - E0) < 1e-6:
            zc = z.copy()
    z = zc if zc is not None else z
    c, OM, B1 = z[:4 * N], z[-2], z[-1]
    X1 = complex(c[0], -c[1])
    print(f"RLC exact zero: E0={E0}, (Om*,b1*)=({OM:.5f},{B1:.5f})  |Y3|={abs(Y3(c)):.2e}  |X1|={abs(X1):.4f}")
    if abs(Y3(c)) > 1e-12 or B1 > 3:
        print("  WARNING: acquisition may be off the clean branch")

    # drive Jacobian d(ReY3,ImY3)/d(Om,E0) and continuation Jacobian d/d(Om,b1)
    h = 1e-6
    def yOmE(Om, e0):
        cc = solve_hb(Om, B1, e0, c); return np.array([cc[IC], cc[IS]])
    def yOmb(Om, b1):
        cc = solve_hb(Om, b1, E0, c); return np.array([cc[IC], cc[IS]])
    Jd = np.column_stack([(yOmE(OM + h, E0) - yOmE(OM - h, E0)) / (2 * h),
                          (yOmE(OM, E0 + h) - yOmE(OM, E0 - h)) / (2 * h)])
    Jc = np.column_stack([(yOmb(OM + h, B1) - yOmb(OM - h, B1)) / (2 * h),
                          (yOmb(OM, B1 + h) - yOmb(OM, B1 - h)) / (2 * h)])
    for lab, J in [("Om,E0", Jd), ("Om,b1", Jc)]:
        sv = np.linalg.svd(J, compute_uv=False)
        print(f"Jacobian d(ReY3,ImY3)/d({lab}): det={np.linalg.det(J):+.3e}  cond={sv[0]/sv[-1]:.3f}  rank={np.linalg.matrix_rank(J,tol=1e-12)}")

    from pathlib import Path
    from hh_antiresonance import figstyle as fs
    fs.use()
    FD = Path(FIGDIR)

    def sweep_b1(gc, half=0.075, npts=51):
        """X3(beta1) over a tight window, warm-started center-out to stay on-orbit."""
        b1s = np.linspace(B1 - half, B1 + half, npts)
        seed = c.copy() if gc == GC else solve_hb(OM, B1, E0, c, gc)
        res = {}
        g = seed.copy()
        for bb in b1s[b1s >= B1]:
            g = solve_hb(OM, bb, E0, g, gc); res[bb] = Y3(g)
        g = seed.copy()
        for bb in b1s[b1s < B1][::-1]:
            g = solve_hb(OM, bb, E0, g, gc); res[bb] = Y3(g)
        return b1s, np.array([res[bb] for bb in b1s])

    # ---------- Figure A: Argand trajectory of X3 through the origin ----------
    # Authored at its placed width: 0.42\textwidth = 2.90 in (no LaTeX rescaling).
    figA, ax = plt.subplots(figsize=fs.column(0.914), constrained_layout=True)
    _, tr2 = sweep_b1(GC)      # two pathways -> through origin
    _, tr1 = sweep_b1(0.0)     # one pathway  -> misses origin
    ax.axhline(0, color="0.82", lw=0.5, zorder=0); ax.axvline(0, color="0.82", lw=0.5, zorder=0)
    ax.plot(tr1.real, tr1.imag, ls="--", color=fs.AMBER, lw=1.3,
            marker="s", ms=3.0, markevery=6, label="one pathway ($g{=}0$)", zorder=2)
    ax.plot(tr2.real, tr2.imag, ls="-", color=fs.NAVY, lw=1.4,
            marker="o", ms=3.0, markevery=6, label="two pathways", zorder=3)
    ax.plot(0, 0, "o", mfc="white", mec="black", mew=1.0, ms=6, zorder=5)
    ax.annotate("computed root", xy=(0, 0), xytext=(26, 22), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="black", shrinkB=4))
    ax.set_xlabel(r"$\mathrm{Re}\,X_3$"); ax.set_ylabel(r"$\mathrm{Im}\,X_3$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=8, handlelength=1.5,
              columnspacing=1.0, handletextpad=0.4)
    fs.grid(ax); fs.panel(ax, "(a)", dx=-0.20)
    ax.margins(0.12)
    fs.save(figA, FD, "rlc_argand_x3")
    print("wrote rlc_argand_x3.{pdf,png}")

    # ---------- Figure B: two-parameter zero map (tight window, on-orbit) ----------
    figB, ax = plt.subplots(figsize=fs.column(0.914), constrained_layout=True)
    noms = np.linspace(OM - 0.010, OM + 0.010, 37); nb1 = np.linspace(B1 - 0.11, B1 + 0.11, 37)
    ci = np.argmin(np.abs(nb1 - B1))
    RE = np.zeros((len(nb1), len(noms))); IM = np.zeros_like(RE); MAG = np.zeros_like(RE)
    for j, om in enumerate(noms):
        gc0 = solve_hb(om, B1, E0, c)          # center of this column, on-orbit
        g = gc0.copy()
        for i in range(ci, len(nb1)):           # upward from center
            g = solve_hb(om, nb1[i], E0, g); y = Y3(g); RE[i, j], IM[i, j], MAG[i, j] = y.real, y.imag, abs(y)
        g = gc0.copy()
        for i in range(ci - 1, -1, -1):         # downward from center
            g = solve_hb(om, nb1[i], E0, g); y = Y3(g); RE[i, j], IM[i, j], MAG[i, j] = y.real, y.imag, abs(y)
    ext = [noms[0], noms[-1], nb1[0], nb1[-1]]
    # Suppressed = light, loud = dark, with the dark end capped at ~60% grey so the
    # solid/dashed null curves stay legible everywhere, in colour and in print.
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    greys_soft = LinearSegmentedColormap.from_list(
        "greys_soft", mpl.colormaps["Greys"](np.linspace(0.0, 0.42, 256)))
    im = ax.imshow(np.log10(MAG + 1e-6), origin="lower", extent=ext, aspect="auto",
                   cmap=greys_soft, vmin=-6.0, vmax=-1.0)
    ax.contour(noms, nb1, RE, levels=[0], colors=[fs.NAVY], linewidths=1.5, linestyles="solid")
    ax.contour(noms, nb1, IM, levels=[0], colors=[fs.VERMILLION], linewidths=1.5, linestyles="dashed")
    ax.plot(OM, B1, **fs.star(markersize=10))
    # Contours labelled where they run: a three-entry legend box covered a
    # quarter of the map, including part of the Im X_3 = 0 branch.
    ax.text(noms[0] + 0.0006, 1.408, r"$\mathrm{Re}\,X_3=0$", color=fs.NAVY,
            fontsize=7, ha="left", va="bottom", rotation=0)
    ax.text(noms[-1] - 0.0006, 1.317, r"$\mathrm{Im}\,X_3=0$", color=fs.VERMILLION,
            fontsize=7, ha="right", va="top")
    ax.annotate("computed root", xy=(OM, B1), xytext=(-14, -20),
                textcoords="offset points", fontsize=7, ha="right", va="top",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.25", shrinkB=5))
    ax.set_xlabel(r"drive frequency $\Omega$"); ax.set_ylabel(r"on-site cubic $\beta_1$")
    cb = figB.colorbar(im, ax=ax, pad=0.02); cb.set_label(r"$\log_{10}|X_3|$", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    fs.panel(ax, "(b)", dx=-0.24)
    fs.save(figB, FD, "rlc_zero_map")
    print("wrote rlc_zero_map.{pdf,png}")


if __name__ == "__main__":
    main()
