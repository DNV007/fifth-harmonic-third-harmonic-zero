"""Does raising the forcing carry the root past the coupling-quintic fold?

Sec. S5 tracks two opposite-index roots under a positive coupling quintic
k5*d^5 and finds them merging near k5* = 0.110, all at fixed F = 0.30 and
kappa = 0.10 with (Omega, beta_1) free.  run_k3_fold_F_free.py showed that the
analogous bound in the linear coupling is NOT a property of the zero set: the
kappa-fold is a curve kappa*(F) with slope ~1.05, and raising F carries the root
straight past the F = 0.30 value.  The quintic bound was left scoped to its
slice for exactly that reason.

This settles it the same way.  For each F we continue the zero in
(Omega, beta_1, k5) at fixed kappa and read the fold as the FIRST turning point
of k5 on leaving the seed (the global argmax is wrong once the locus runs far).
If k5*(F) rises with F, the few-percent constitutive bound quoted for the
quintic is a slice statement and the admissible quintic grows with drive.

The residual is a JAX reimplementation of the numpy one in
run_k3_quintic_regularized_root.py, so it is validated against that script's
published roots before any sweep is run.

Run:  PYTHONPATH=src uv run python scripts/run_k3_quintic_fold_F.py
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, root

jax.config.update("jax_enable_x64", True)

from hh_antiresonance.continuation import (
    bordered_newton_corrector, moore_penrose_tangent, orient_tangent,
)

DATADIR = Path("data"); DATADIR.mkdir(exist_ok=True)

W1, W2 = 1.0, 1.25
Z1, Z2 = 0.015, 0.02
KA, KNL = 0.10, -0.30
NH, NT = 7, 512
NC = 4 * NH
IC, IS = 2 * (3 - 1), 2 * (3 - 1) + 1          # driven-mass 3rd harmonic
I1C, I1S = 0, 1                                 # driven-mass fundamental

_n = np.arange(NT)
_PH = 2 * np.pi * np.outer(np.arange(1, NH + 1), _n) / NT
COS, SIN = jnp.asarray(np.cos(_PH)), jnp.asarray(np.sin(_PH))
MV = jnp.asarray(np.arange(1, NH + 1)[:, None])

# k5 continuation range guard: the tracked family sits near Omega ~ 0.26
OM_WINDOW = (0.20, 0.35)


def log(m): print(m, flush=True)


def _recon(c, Om):
    x1 = c[:2 * NH].reshape(NH, 2); x2 = c[2 * NH:].reshape(NH, 2)
    a1, b1 = x1[:, [0]], x1[:, [1]]
    a2, b2 = x2[:, [0]], x2[:, [1]]
    q1 = jnp.sum(a1 * COS + b1 * SIN, 0); q2 = jnp.sum(a2 * COS + b2 * SIN, 0)
    v1 = jnp.sum(MV * Om * (-a1 * SIN + b1 * COS), 0)
    v2 = jnp.sum(MV * Om * (-a2 * SIN + b2 * COS), 0)
    ac1 = jnp.sum(-(MV * Om) ** 2 * (a1 * COS + b1 * SIN), 0)
    ac2 = jnp.sum(-(MV * Om) ** 2 * (a2 * COS + b2 * SIN), 0)
    return q1, v1, ac1, q2, v2, ac2


def _hb(c, Om, b1, F, k5):
    q1, v1, ac1, q2, v2, ac2 = _recon(c, Om)
    d = q1 - q2
    cpl = KA * d + KNL * d ** 3 + k5 * d ** 5
    r1 = ac1 + 2 * Z1 * v1 + W1 ** 2 * q1 + b1 * q1 ** 3 + cpl - F * COS[0]
    r2 = ac2 + 2 * Z2 * v2 + W2 ** 2 * q2 - cpl
    proj = lambda r: jnp.concatenate([(2 / NT) * (r @ COS.T)[:, None],
                                      (2 / NT) * (r @ SIN.T)[:, None]], axis=1)
    P1, P2 = proj(r1), proj(r2)
    return jnp.concatenate([P1, P2], axis=1).reshape(-1)   # [c1,s1,c2,s2] per m


def _res_fixed(z, F, k5):
    c = z[:NC]
    hb = _hb(c, z[NC], z[NC + 1], F, k5)
    return jnp.concatenate([hb, jnp.array([c[IC], c[IS]], dtype=hb.dtype)])


def _res_freek5(z, F):
    c = z[:NC]
    hb = _hb(c, z[NC], z[NC + 1], F, z[NC + 2])
    return jnp.concatenate([hb, jnp.array([c[IC], c[IS]], dtype=hb.dtype)])


_rf = jax.jit(_res_fixed); _jf = jax.jit(jax.jacfwd(_res_fixed, argnums=0))
_rk = jax.jit(_res_freek5); _jk = jax.jit(jax.jacfwd(_res_freek5, argnums=0))
res_fixed = lambda z, F, k5: np.asarray(_rf(jnp.asarray(z, float), float(F), float(k5)), float)
jac_fixed = lambda z, F, k5: np.asarray(_jf(jnp.asarray(z, float), float(F), float(k5)), float)


_rc = jax.jit(lambda c, Om, b1, F, k5: _hb(c, Om, b1, F, k5))
_jc = jax.jit(jax.jacfwd(lambda c, Om, b1, F, k5: _hb(c, Om, b1, F, k5), argnums=0))


def solve_hb(Om, b1, F, k5, c0=None):
    """Plain harmonic balance at fixed (Omega, beta_1): seeds the augmented solve."""
    if c0 is None:
        c0 = np.zeros(NC); c0[0] = F / (W1 ** 2 - Om ** 2)
    f = lambda c: np.asarray(_rc(jnp.asarray(c, float), float(Om), float(b1),
                                 float(F), float(k5)), float)
    J = lambda c: np.asarray(_jc(jnp.asarray(c, float), float(Om), float(b1),
                                 float(F), float(k5)), float)
    s = root(f, np.asarray(c0, float), jac=J, method="hybr", tol=1e-14,
             options={"xtol": 1e-14, "maxfev": 4000})
    if s.success:
        return s.x
    return least_squares(f, np.asarray(c0, float), jac=J,
                         xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=6000).x


def solve_zero(Om0, b0, F, k5, c0=None):
    # Seed the augmented system with a converged orbit at (Om0, b0); without
    # this the least-squares solve can wander off to the trivial Omega -> 0 sheet.
    c0 = solve_hb(Om0, b0, F, k5, c0)
    z0 = np.concatenate([np.asarray(c0, float), [Om0, b0]])
    s = least_squares(lambda z: res_fixed(z, F, k5), z0,
                      jac=lambda z: jac_fixed(z, F, k5),
                      xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
    s2 = root(lambda z: res_fixed(z, F, k5), s.x, jac=lambda z: jac_fixed(z, F, k5),
              method="hybr", tol=1e-15, options={"xtol": 1e-15, "maxfev": 4000})
    z = s2.x if (np.linalg.norm(res_fixed(s2.x, F, k5))
                 < np.linalg.norm(res_fixed(s.x, F, k5))) else s.x
    c = z[:NC]
    return dict(Om=float(z[NC]), b1=float(z[NC + 1]), coeffs=c,
                X3=float(np.hypot(c[IC], c[IS])),
                X1=float(np.hypot(c[I1C], c[I1S])))


def k5_fold_at(F, seed, k5_seed=0.0, nsteps=500):
    """Continue in (Omega, beta_1, k5); return the first turning point of k5."""
    res = lambda z: np.asarray(_rk(jnp.asarray(z, float), float(F)), float)
    jac = lambda z: np.asarray(_jk(jnp.asarray(z, float), float(F)), float)
    state0 = np.concatenate([seed["coeffs"], [seed["Om"], seed["b1"], k5_seed]])

    def run(direction):
        st = state0.copy()
        t, _ = moore_penrose_tangent(jac(st))
        t = direction * orient_tangent(t, preferred_index=-1)
        S = [st.copy()]; ds = 2e-3; dsmin, dsmax = 1e-7, 8e-3
        for _ in range(nsteps):
            ok = False
            for _try in range(40):
                pred = st + ds * t
                try:
                    cor, info = bordered_newton_corrector(
                        res, pred, t, jac_func=jac, tol=1e-10, max_iter=40)
                except Exception:
                    info = {"converged": False}; cor = pred
                if info["converged"]:
                    ok = True; break
                ds *= 0.5
                if ds < dsmin:
                    break
            if not ok or cor[NC] < 0.05:
                break
            try:
                t, _ = moore_penrose_tangent(jac(cor), reference_tangent=t)
            except Exception:
                sec = cor - st; t = sec / (np.linalg.norm(sec) + 1e-14)
            st = cor; S.append(st.copy())
            ds = min(dsmax, 1.4 * ds) if int(info.get("iters", 3)) < 4 else max(dsmin, 0.5 * ds)
        return np.asarray(S)

    def first_turn(path):
        k = path[:, NC + 2]
        if len(k) < 3 or k[1] <= k[0]:
            return None
        for i in range(1, len(k) - 1):
            if k[i + 1] < k[i]:
                return i
        return None

    best = None
    for d in (+1, -1):
        p = run(d); i = first_turn(p)
        if i is None:
            continue
        if best is None or p[i, NC + 2] > best[0][best[1], NC + 2]:
            best = (p, i)
    if best is None:
        raise RuntimeError("no k5 turning point found")
    loc, i = best
    Om = float(loc[i, NC])
    return dict(F=float(F), k5star=float(loc[i, NC + 2]), Om=Om,
                b1=float(loc[i, NC + 1]),
                X3=float(np.hypot(loc[i, IC], loc[i, IS])),
                X1=float(np.hypot(loc[i, I1C], loc[i, I1S])),
                npts=len(loc), on_family=bool(OM_WINDOW[0] <= Om <= OM_WINDOW[1]),
                coeffs=loc[i, :NC].copy())


def main():
    log("=== does raising F carry the root past the quintic fold? ===")
    log(f"fixed: kappa={KA}, kappa_nl={KNL}, N_H={NH}\n")

    # ---- (0) validate the JAX residual against the published numpy roots ----
    log("(0) validation against run_k3_quintic_regularized_root.py:")
    for k5, oe, be in ((0.00, 0.2597281, 0.2381543), (0.08, 0.2578800, 0.2322264)):
        s = solve_zero(oe, be, 0.30, k5)
        log(f"  k5={k5:.2f}: Omega*={s['Om']:.7f} (published {oe})  "
            f"beta1*={s['b1']:.7f} (published {be})  |X3|={s['X3']:.1e}")
        assert abs(s["Om"] - oe) < 2e-6 and abs(s["b1"] - be) < 2e-6, "residual mismatch"
    log("  residual reproduces both published roots.\n")

    # ---- (1) seeds at k5 = 0 for each F -------------------------------------
    F_GRID = [0.28, 0.30, 0.32, 0.34, 0.36, 0.40, 0.45]
    log("(1) k5=0 seed roots vs F:")
    log(f"{'F':>6} {'Omega*':>12} {'beta1*':>12} {'|X3|':>10}")
    seeds = {}
    prev = dict(Om=0.2597281, b1=0.2381543, coeffs=None)
    for F in sorted(F_GRID, key=lambda x: abs(x - 0.30)):
        s = solve_zero(prev["Om"], prev["b1"], F, 0.0, c0=prev["coeffs"])
        if s["X3"] > 1e-9 or not (OM_WINDOW[0] <= s["Om"] <= OM_WINDOW[1]):
            log(f"{F:6.2f}   no root on the family"); continue
        seeds[F] = s
        if abs(F - 0.30) < 0.05:
            prev = s
        log(f"{F:6.2f} {s['Om']:12.7f} {s['b1']:12.7f} {s['X3']:10.1e}")

    # ---- (2) k5 fold at each F ---------------------------------------------
    log("\n(2) k5-fold from (Omega,beta_1,k5) continuation at each F:")
    log(f"{'F':>6} {'k5*(F)':>10} {'Omega':>11} {'beta1':>11} {'|X3|':>9} "
        f"{'pts':>5}  {'quintic/cubic':>13}")
    rows = []
    for F in sorted(seeds):
        try:
            r = k5_fold_at(F, seeds[F])
        except Exception as e:
            log(f"{F:6.2f}   continuation failed: {type(e).__name__}"); continue
        if not r["on_family"]:
            log(f"{F:6.2f} {r['k5star']:10.5f} {r['Om']:11.6f}   <-- off family, discarded")
            continue
        rows.append(r)
        log(f"{F:6.2f} {r['k5star']:10.5f} {r['Om']:11.6f} {r['b1']:11.6f} "
            f"{r['X3']:9.1e} {r['npts']:5d}")

    if not rows:
        log("\nno usable rows"); return

    Fs = np.array([r["F"] for r in rows]); ks = np.array([r["k5star"] for r in rows])
    log(f"\n  k5*(F) runs {ks.min():.4f} -> {ks.max():.4f} over F in "
        f"[{Fs.min():.2f},{Fs.max():.2f}]")
    log(f"  monotone increasing: {bool(np.all(np.diff(ks) > 0))};  "
        f"mean slope d(k5*)/dF = {np.polyfit(Fs, ks, 1)[0]:.3f}")

    # ---- (3) direct roots beyond the F=0.30 quintic fold -------------------
    K_REF = ks[np.argmin(np.abs(Fs - 0.30))] if np.any(np.abs(Fs - 0.30) < 1e-9) else None
    log(f"\n(3) augmented solves at k5 beyond the F=0.30 fold"
        + (f" ({K_REF:.5f})" if K_REF else "") + ":")
    log(f"{'k5':>7} {'F':>6} {'Omega*':>12} {'beta1*':>12} {'|X3|':>10} {'|X1|':>8}")
    checks = []
    for k5t in (0.12, 0.15, 0.18, 0.22):
        cand = [r for r in rows if r["k5star"] > k5t + 2e-3]
        if not cand:
            log(f"{k5t:7.3f}   no F on the grid has k5*(F) > {k5t}"); continue
        r = min(cand, key=lambda q: q["F"])
        s = solve_zero(r["Om"], r["b1"], r["F"], k5t, c0=r["coeffs"])
        ok = s["X3"] < 1e-9 and OM_WINDOW[0] <= s["Om"] <= OM_WINDOW[1]
        checks.append(dict(k5=k5t, F=r["F"], **{k: s[k] for k in ("Om", "b1", "X3", "X1")}, ok=ok))
        log(f"{k5t:7.3f} {r['F']:6.2f} {s['Om']:12.7f} {s['b1']:12.7f} "
            f"{s['X3']:10.1e} {s['X1']:8.4f}" + ("" if ok else "   <-- not a root"))

    log("\n(4) verdict:")
    surv = [c for c in checks if c["ok"]]
    if surv and bool(np.all(np.diff(ks) > 0)):
        log(f"  k5*(F) increases with drive, and zeros persist to k5={max(c['k5'] for c in surv):.2f}")
        log(f"  at F={max(c['F'] for c in surv):.2f}. The quintic bound behaves exactly like the")
        log("  coupling fold: it is a property of the fixed-F slice, not a global")
        log("  constitutive tolerance. The few-percent force fraction quoted at")
        log("  F=0.30 should be read as a slice value that grows with drive.")
    else:
        log("  k5*(F) does not increase monotonically on this grid; the quintic")
        log("  bound does not simply track the drive. Report the slice value.")

    np.savetxt(DATADIR / "two_dof_k3_quintic_fold_F.csv",
               np.array([[r["F"], r["k5star"], r["Om"], r["b1"], r["X1"], r["X3"]] for r in rows]),
               delimiter=",", header="F,k5_star,omega_fold,beta1_fold,abs_X1,abs_X3",
               comments="")
    log(f"\nwrote {DATADIR/'two_dof_k3_quintic_fold_F.csv'}")


if __name__ == "__main__":
    main()
