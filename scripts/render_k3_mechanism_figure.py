"""Mechanism figure: how the third-harmonic zero is closed, and that it is not a low-Q effect.

Three panels, all computed here from the converged orbit and from the absorber-loss
continuation of run_k3_high_q_continuation.py:

  (a) the projected third-harmonic balance as complex vectors.  The
      fundamental-only source leaves a residual; the 3*Omega feedback removes
      most of its magnitude; the 5*Omega feedback supplies the remaining phase
      direction; the sum closes.
  (b) the approximation hierarchy {1}, {1,3}, {1,3,5}: relative error in the
      closing cubic coefficient and the residual phase of Xi that each leaves.
The homotopy that used to be panel (c) now leads Figure 1, beside the N_H = 3
and N_H = 7 quadrature maps, where it belongs: those three panels are the
argument in order.  What remains here is how the closure is apportioned.

The common-damping continuation that used to occupy panel (c) is secondary
robustness evidence rather than mechanism evidence; it is rendered separately
here as k3_high_q.pdf for the Supplement.

Run:  PYTHONPATH=src uv run python scripts/render_k3_mechanism_figure.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hh_antiresonance import figstyle as fs

fs.use()

import run_k3_high_q_continuation as hq

_D = Path(os.path.dirname(os.path.abspath(__file__)))
DATA = _D / ".." / "data"
FIGS = [_D / ".." / "figures"]

OM_STAR, B1_STAR = 0.25972810, 0.23815428
N = hq.N


def source_vectors():
    """Projected balance of Eq. (S10) on the converged orbit, by source truncation."""
    seed = np.concatenate([hq.solve_hb(OM_STAR, B1_STAR, hq.Z2_BASE), [OM_STAR, B1_STAR]])
    z, _ = hq.solve_zero(seed, hq.Z2_BASE)
    c, Om, b1 = z[:4 * N], float(z[-2]), float(z[-1])
    U1, U2 = hq.U_of(c, 1), hq.U_of(c, 2)
    Dv = U1 - U2

    def LHS(keep):
        return (hq.Z22(3, Om, hq.Z2_BASE) * b1 * hq.A3_multinomial(U1, keep)
                + hq.L2(3, Om, hq.Z2_BASE) * hq.KNL * hq.A3_multinomial(Dv, keep))

    l1, l13, l135 = LHS((1,)), LHS((1, 3)), LHS((1, 3, 5))
    dec = hq.decompose(c, Om, b1, hq.Z2_BASE)
    return l1, l13 - l1, l135 - l13, l135, b1, dec


def main():
    l1, c3, c5, closed, b1, dec = source_vectors()
    # Normalize to the fundamental-only term AND rotate so that term lies along
    # the real axis. Both are conventions of presentation: the balance is
    # homogeneous, so only the relative lengths and angles carry meaning, and in
    # the unrotated frame the three vectors sit within 8 degrees of one axis,
    # which hides the very geometry the panel exists to show.
    rot = np.exp(-1j * np.angle(l1)) / abs(l1)
    l1n, c3n, c5n = l1 * rot, c3 * rot, c5 * rot

    fig = plt.figure(figsize=fs.full(0.30))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0],
                          left=0.075, right=0.945, bottom=0.175, top=0.90,
                          wspace=0.42)
    ax = [fig.add_subplot(gs[0, i]) for i in range(2)]

    # ---------------- (a) closure of the projected source vectors -----------
    a_ax = ax[0]

    def chain(axis, vecs, lw=1.4, ms=9, origin=0j):
        tip = origin
        for v, i in vecs:
            col, ls, _ = fs.SERIES[i]
            axis.annotate("", xy=(np.real(tip + v), np.imag(tip + v)),
                          xytext=(np.real(tip), np.imag(tip)),
                          arrowprops=dict(arrowstyle="-|>", color=col, linestyle=ls,
                                          linewidth=lw, shrinkA=0, shrinkB=0,
                                          mutation_scale=ms, joinstyle="miter"))
            tip = tip + v
        return tip

    vecs = [(l1n, 0), (c3n, 1), (c5n, 2)]
    chain(a_ax, vecs)
    a_ax.plot([0], [0], **fs.star(markersize=7))
    a_ax.text(0.04, 0.022, r"$\{1\}$ source", color=fs.SERIES[0][0], fontsize=7.5,
              ha="left", va="bottom")
    a_ax.text(0.45, -0.098, r"$3\Omega$ feedback", color=fs.SERIES[1][0], fontsize=7.5,
              ha="center", va="top")
    a_ax.set_xlabel("Re, in units of the $\\{1\\}$ term")
    a_ax.set_ylabel("Im")
    a_ax.set_aspect("equal", adjustable="box")
    a_ax.set_xlim(-0.20, 1.15)
    a_ax.set_ylim(-0.40, 0.56)
    a_ax.set_xticks([0.0, 0.5, 1.0])
    a_ax.set_yticks([-0.25, 0.0, 0.25, 0.5])
    fs.grid(a_ax)

    # Zoom on the closure. The two large terms are nearly anti-parallel, so what
    # they leave is seven times shorter than either of them, and the closing
    # 5*Omega vector cannot be read at the scale of the panel. No connector is
    # drawn: it would have to cross the chain it magnifies.
    r13 = l1n + c3n
    ctr = 0.5 * r13
    half = 0.115
    INSET_W = 0.47
    ins = a_ax.inset_axes([0.40, 0.55, INSET_W, 0.40])
    ins.plot([np.real(l1n), np.real(r13)], [np.imag(l1n), np.imag(r13)],
             color=fs.SERIES[1][0], linestyle=fs.SERIES[1][1], lw=1.1, clip_on=True)
    ins.annotate("", xy=(0, 0), xytext=(np.real(r13), np.imag(r13)),
                 arrowprops=dict(arrowstyle="-|>", color=fs.SERIES[2][0],
                                 linestyle=fs.SERIES[2][1], linewidth=1.2,
                                 shrinkA=0, shrinkB=0, mutation_scale=7))
    ins.plot([0], [0], **fs.star(markersize=5))
    ins.set_xlim(np.real(ctr) - half, np.real(ctr) + half)
    ins.set_ylim(np.imag(ctr) - half * 0.60, np.imag(ctr) + half * 0.60)
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_facecolor("white")
    for sp in ins.spines.values():
        sp.set_linewidth(0.5); sp.set_color("0.5")
    ins.text(0.05, 0.93, r"$5\Omega$", color=fs.SERIES[2][0], fontsize=7.5,
             ha="left", va="top", transform=ins.transAxes)
    x0, x1 = a_ax.get_xlim()
    mag = INSET_W * (x1 - x0) / (2 * half)
    ins.text(0.97, 0.06, rf"$\times{mag:.1f}$", color="0.35", fontsize=6.5,
             ha="right", va="bottom", transform=ins.transAxes)

    fs.panel(a_ax, "(a)", dx=-0.06, dy=1.03)

    # ---------------- (b) what each source truncation leaves ----------------
    b_ax = ax[1]
    labels = [r"$\{1\}$", r"$\{1,3\}$", r"$\{1,3,5\}$"]
    err_b1 = [abs(dec["b1_1"] / b1 - 1), abs(dec["b1_13"] / b1 - 1),
              abs(dec["b1_135"] / b1 - 1)]
    phase = [abs(dec["argLO"]), abs(dec["argLO"] + dec["im3"]),
             abs(dec["argLO"] + dec["im3"] + dec["im5"])]
    x = np.arange(3); w = 0.36
    b_ax.bar(x - w / 2, err_b1, w, color=fs.NAVY, edgecolor="black", linewidth=0.4,
             label=r"$|\Delta\beta_1|/\beta_1^\ast$")
    b_ax.bar(x + w / 2, phase, w, color=fs.AMBER, edgecolor="black", linewidth=0.4,
             hatch="///", label=r"$|\arg\Xi|$")
    b_ax.set_yscale("log")
    b_ax.set_ylim(1e-7, 4e-1)
    b_ax.set_xticks(x); b_ax.set_xticklabels(labels)
    b_ax.set_xlabel("harmonics kept in the source")
    b_ax.set_ylabel("residual magnitude and phase")
    b_ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=2,
                fontsize=7, frameon=False, handlelength=1.2, columnspacing=1.0,
                handletextpad=0.5)
    fs.grid(b_ax)
    fs.panel(b_ax, "(b)", dx=-0.30, dy=1.03)

    # ---------------- (c) the same split continued to high Q ----------------
    for d in FIGS:
        d.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS[0] / "k3_mechanism.pdf")
    fig.savefig(FIGS[0] / "k3_mechanism.png", dpi=400)

    # ---- the common-damping continuation, now a Supplement figure ---------
    figq, q_ax = plt.subplots(figsize=fs.column(0.78), constrained_layout=True)
    R = np.loadtxt(DATA / "two_dof_k3_common_damping.csv", delimiter=",", skiprows=1)
    Q2, lw = R[:, 3], R[:, 10]
    phase_q = np.abs(R[:, 12])
    mag_q = np.abs(R[:, 15] / R[:, 5] - 1)
    col0, ls0, mk0 = fs.SERIES[0]
    col1, ls1, mk1 = fs.SERIES[1]
    q_ax.loglog(Q2, mag_q, color=col1, linestyle=ls1, marker=mk1, markersize=2.4,
                markevery=3)
    q_ax.loglog(Q2, phase_q, color=col0, linestyle=ls0, marker=mk0, markersize=2.4,
                markevery=3)
    q_ax.set_xlabel(r"quality factor $Q_2$")
    q_ax.set_ylabel("deficit left by the fundamental")
    q_ax.set_xlim(20, 8e5)
    q_ax.set_ylim(1.2e-7, 3e-1)
    q_ax.text(3.5e2, 5.5e-2, "magnitude", color=col1, fontsize=7, ha="center")
    q_ax.text(2.2e4, 4.5e-4, "phase", color=col0, fontsize=7, ha="left")
    fs.grid(q_ax)
    q2 = q_ax.twinx()
    q2.loglog(Q2, lw, color=fs.REFERENCE, linestyle=":", linewidth=1.0)
    q2.set_ylabel(r"half-widths from $\omega_+$", color=fs.REFERENCE, fontsize=7.5)
    q2.tick_params(axis="y", colors=fs.REFERENCE, labelsize=7)
    q2.set_ylim(3e-2, 3e4)
    q2.axhline(1.0, color=fs.REFERENCE, lw=0.4, ls=(0, (1, 3)))
    q2.text(1.1e3, 3.0e2, r"$5\Omega^\ast$ detuning", color=fs.REFERENCE,
            fontsize=7, ha="center")
    q_ax.set_zorder(q2.get_zorder() + 1); q_ax.patch.set_visible(False)
    figq.savefig(FIGS[0] / "k3_high_q.pdf")
    figq.savefig(FIGS[0] / "k3_high_q.png", dpi=400)

    print("panel (a) vectors (normalized to the leading residual):")
    print(f"  fundamental-only residual |LHS(1)| = 1  (raw {abs(l1):.4e})")
    print(f"  3W channel  {abs(c3)*1/abs(l1):.4f} at "
          f"{np.degrees(np.angle(c3/l1))%360:.1f} deg to it")
    print(f"  5W channel  {abs(c5)*1/abs(l1):.4f}")
    print(f"  closed sum  {abs(closed)/abs(l1):.2e}")
    print("panel (b):")
    for lab, e, p in zip(("{1}", "{1,3}", "{1,3,5}"), err_b1, phase):
        print(f"  {lab:<9} |db1|/b1 = {e:.3e}   residual |arg Xi| = {p:.3e}")
    print("k3_high_q (Supplement), common damping: phase deficit %.2e -> %.2e, "
          "magnitude deficit %.3f%% -> %.3f%%, detuning %.2f -> %.0f half-widths"
          % (phase_q[0], phase_q[-1], mag_q[0] * 100, mag_q[-1] * 100, lw[0], lw[-1]))
    print(f"wrote {FIGS[0]/'k3_mechanism.pdf'} and {FIGS[0]/'k3_high_q.pdf'}")


if __name__ == "__main__":
    main()
