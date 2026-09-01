"""Wide-window scan of the leading-order (fundamental-only) closure condition.

The Letter's leading-order object is

    Xi_LO(Omega) = -kappa_nl L2(3W)/Z22(3W) (1 - kappa/Z22(W))^3,

a ratio of linear propagators.  A leading-order zero requires beta_1 = Xi_LO
with beta_1 real, i.e. Im Xi_LO = 0.  The Letter states that absorber loss
leaves a phase that does not vanish; that statement must be LOCAL, because the
perturbative comparison branch of the Supplement sits at a genuine crossing
with beta_1 < 0.

This driver enumerates every crossing of Im Xi_LO on a wide frequency window,
classifies each one, and repeats the scan as zeta_2 is reduced, quantifying the
sense in which the phase obstruction is set by absorber loss.

Reported:
  (1) all zeros of Im Xi_LO on Omega in (0.02, 3.0) at the baseline zeta_2,
      a window chosen to contain the whole traced zero locus of Sec. S6 B,
      which reaches Omega = 2.174,
      with Re Xi_LO (= the required beta_1) and a classification
  (2) the phase over the window containing the reported branch, and the margin
      by which it fails to cross
  (3) the same scan at reduced zeta_2: the crossing structure is unchanged in
      count, while the phase amplitude away from the crossings scales with
      absorber loss

Run:  uv run python scripts/run_k3_leading_order_scan.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

W1, W2 = 1.0, 1.25
Z1 = 0.015
KA, KNL = 0.10, -0.30
Z2_BASE = 0.02
OM_STAR = 0.25972810           # reported finite-amplitude root
OM_PERT = 1.24849746           # perturbative comparison branch, F -> 0

_D = Path(os.path.dirname(os.path.abspath(__file__)))
OUT = _D / ".." / "data"


def L2(k, Om, z2):
    kO = k * Om
    return W2 ** 2 - kO ** 2 + 2j * z2 * kO


def Z22(k, Om, z2):
    return L2(k, Om, z2) + KA


def xi_lo(Om, z2=Z2_BASE):
    return -KNL * L2(3, Om, z2) / Z22(3, Om, z2) * (1.0 - KA / Z22(1, Om, z2)) ** 3


def im_xi(Om, z2=Z2_BASE):
    return float(np.imag(xi_lo(Om, z2)))


def crossings(z2, lo=0.02, hi=3.0, npts=7500001):
    """All sign changes of Im Xi_LO.  The grid is fine enough to resolve the
    crossings that sit inside the coupled-mode poles, whose width is set by
    zeta_2: a coarse grid loses them as the loss is reduced, so the crossing
    count is a property of the grid there, not of the physics."""
    g = np.linspace(lo, hi, npts)
    v = np.imag(xi_lo(g, z2))
    out = []
    sgn = np.sign(v)
    for i in np.nonzero(np.diff(sgn) != 0)[0]:
        try:
            r = brentq(im_xi, g[i], g[i + 1], args=(z2,), xtol=1e-14, rtol=1e-15)
        except ValueError:
            continue
        out.append((float(r), float(np.real(xi_lo(r, z2)))))
    return out


def poles(z2):
    """Frequencies at which Xi_LO is singular or its numerator vanishes."""
    K = np.array([[W1 ** 2 + KA, -KA], [-KA, W2 ** 2 + KA]])
    wm, wp = np.sqrt(np.sort(np.linalg.eigvals(K).real))
    return {"3W = w_-": wm / 3, "3W = w_+": wp / 3, "W = w_-": wm, "W = w_+": wp,
            "3W = w_2 (L2 zero)": W2 / 3}


def classify(Om, beta, z2):
    """Label a leading-order crossing by what it is attached to."""
    if abs(beta) > 10.0:
        return "|Xi_LO| divergent: required cubic unbounded, physically irrelevant"
    if abs(Om - W2) < 5e-3 and abs(beta) < 1e-3:
        return "collapses onto the linear absorber tuning (Omega -> omega_2, beta_1 -> 0)"
    if abs(Om - OM_PERT) < 5e-3:
        return "perturbative comparison branch (SM Sec. S10)"
    for name, w in poles(z2).items():
        if abs(Om - w) < max(30 * z2, 2e-3):
            return f"inside the {name} resonance (width ~ zeta_2)"
    sign = "softening" if beta < 0 else "hardening"
    return f"{sign} on-site cubic required (beta_1 = {beta:+.4f})"


def main():
    hdr = "=" * 100
    print(hdr)
    print("LEADING-ORDER (FUNDAMENTAL-ONLY) CLOSURE CONDITION: WIDE-WINDOW SCAN")
    print(hdr)
    print(f"fixed: omega1={W1}, omega2={W2}, zeta1={Z1}, kappa={KA}, kappa_nl={KNL}")
    print(f"the reported finite-amplitude root sits at Omega* = {OM_STAR:.6f}, "
          f"beta_1* = +0.238154\n")

    print(f"(1) ZEROS OF Im Xi_LO ON Omega in (0.02, 3.0) at zeta_2 = {Z2_BASE}")
    cr = crossings(Z2_BASE)
    print(f"    {'Omega':>10}{'Re Xi_LO':>12}{'|Xi_LO|':>11}   classification")
    for om, be in cr:
        print(f"    {om:>10.6f}{be:>12.6f}{abs(xi_lo(om)):>11.4f}   "
              f"{classify(om, be, Z2_BASE)}")
    print(f"    total crossings: {len(cr)}")

    # Omega = 0 is a boundary root of Im Xi_LO -- the scan window cannot see it,
    # and Xi_LO(0) is real and positive, close to the beta_1 of the reported
    # branch. That route is closed instead by the slope: Im Xi_LO = c0*Omega +
    # O(Omega^3) with c0 > 0, so there is no crossing at any small Omega > 0.
    print(f"\n(1b) THE Omega -> 0 BOUNDARY")
    print(f"    Xi_LO(0) = {xi_lo(0.0).real:.12f} (real; Im = {xi_lo(0.0).imag:.1e})")
    print(f"    {'Omega':>10}{'Im Xi_LO':>16}{'Im Xi_LO / Omega':>20}")
    for o_ in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        print(f"    {o_:>10.0e}{im_xi(o_):>16.6e}{im_xi(o_) / o_:>20.8f}")
    c0 = im_xi(1e-6) / 1e-6
    print(f"    c0 = lim Im Xi_LO/Omega = {c0:.8f} > 0, so Im Xi_LO has no root")
    print(f"    at any small Omega > 0 and the leading-order phase obstruction")
    print(f"    survives down to zero frequency. The joint (F,Omega)->(0,0)")
    print(f"    corner is a two-variable limit this scan does not address.")
    assert c0 > 0, "c0 <= 0: the Omega -> 0 route is NOT closed"
    near = [c for c in cr if abs(c[0] - OM_STAR) < 0.10]
    print(f"    crossings within 0.10 of Omega*: {len(near)}")

    print(f"\n(2) PHASE OVER THE WINDOW OF THE REPORTED BRANCH")
    win = np.linspace(0.20, 0.34, 15)
    print(f"    {'Omega':>8}{'arg Xi_LO':>13}{'Im Xi_LO':>13}{'Re Xi_LO':>12}")
    for o in win:
        x = xi_lo(o)
        print(f"    {o:>8.3f}{np.angle(x):>13.4e}{x.imag:>13.4e}{x.real:>12.6f}")
    ph = np.array([np.angle(xi_lo(o)) for o in np.linspace(0.20, 0.34, 1401)])
    print(f"    sign changes on [0.20, 0.34]: {int(np.sum(np.diff(np.sign(ph)) != 0))}"
          f"   monotone increasing: {bool(np.all(np.diff(ph) > 0))}")
    print(f"    phase stays positive; minimum over the window = {ph.min():.3e} rad")

    print(f"\n(3) THE SAME SCAN AS ABSORBER LOSS IS REDUCED")
    print(f"    {'zeta_2':>10}{'Q_2':>9}{'crossings':>11}{'nearest to Om*':>16}"
          f"{'arg Xi_LO(Om*)':>16}{'ratio to zeta_2':>17}")
    rows = []
    detail = {}
    for z2 in (2e-2, 5e-3, 1e-3, 1e-4, 1e-5, 2e-6):
        cz = crossings(z2)
        detail[z2] = cz
        d = min(abs(c[0] - OM_STAR) for c in cz)
        a = np.angle(xi_lo(OM_STAR, z2))
        print(f"    {z2:>10.1e}{1/(2*z2):>9.0f}{len(cz):>11d}{d:>16.4f}"
              f"{a:>16.3e}{a/z2:>17.4f}")
        rows.append((z2, 1 / (2 * z2), len(cz), d, a, a / z2))
    print("\n    where those crossings sit:")
    for z2, cz in detail.items():
        print(f"      zeta_2 = {z2:.0e}:")
        for om, be in cz:
            print(f"        Omega = {om:>10.6f}  beta_1 = {be:>11.4f}   "
                  f"{classify(om, be, z2)}")
    print("\n    Every crossing found at any loss sits either on the perturbative")
    print("    comparison branch (which collapses onto the linear absorber tuning")
    print("    Omega -> omega_2, beta_1 -> 0 as the loss is reduced) or at a")
    print("    divergence of |Xi_LO| inside the upper-mode resonance, where the")
    print("    required cubic is unbounded; the nearest one to the reported root stays")
    print("    about one unit away in Omega at every loss tested.  The reported")
    print("    branch therefore never acquires a leading-order root at its own")
    print("    frequency.  What changes with loss is the size of the phase there,")
    print("    which is proportional to zeta_2 (last column, constant to 4 digits).")

    OUT.mkdir(exist_ok=True)
    np.savetxt(OUT / "two_dof_k3_leading_order_scan.csv", np.array(rows), delimiter=",",
               comments="",
               header="zeta2,Q2,n_crossings,dist_to_omega_star,arg_xi_lo_at_star,"
                      "arg_over_zeta2")
    np.savetxt(OUT / "two_dof_k3_leading_order_crossings.csv", np.array(cr), delimiter=",",
               comments="", header="omega_crossing,re_xi_lo")
    print(f"\nwrote {OUT/'two_dof_k3_leading_order_scan.csv'}")
    print(f"wrote {OUT/'two_dof_k3_leading_order_crossings.csv'}")


if __name__ == "__main__":
    main()
