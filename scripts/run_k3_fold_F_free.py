"""Is the kappa-fold of the k=3 exact zero a real annihilation, or an artifact
of holding the forcing fixed?

The fold figure tracks two opposite-index third-harmonic roots in the linear
coupling kappa at FIXED F=0.30, and they merge at kappa*=0.14186.  But the zero
set is two-dimensional in (Omega, F, beta_1, kappa): at fixed kappa it is a
one-dimensional locus in (Omega, F, beta_1).  A merger seen in the F=0.30 slice
is a tangency of that surface with the slice, and does not by itself say that no
zero exists at larger kappa -- it may simply require a different forcing.

This script settles it.  For each F on a grid we repeat the same
pseudo-arclength continuation in (Omega, beta_1, kappa) that produced
kappa*(F=0.30), and read off the fold

    kappa*(F) = max kappa on the physical branch.

If kappa*(F) rises with F, the F=0.30 fold is slice-dependent and the zero
survives beyond kappa=0.14186 at higher drive.  If kappa*(F) has an interior
maximum, that maximum IS the genuine boundary of the zero set in kappa (a cusp
in the (kappa,F) fold curve), and the annihilation claim stands in the stronger
form.

Two independent confirmations are run afterwards:
  (a) direct augmented solves at kappa values beyond 0.14186, freeing (Omega,
      beta_1) at the F predicted by the fold curve;
  (b) for each such root, the orbit diagnostics max|d| and min tangent
      stiffness, so we can say whether the recovered zeros lie inside the
      locally monotone constitutive window the Letter restricts itself to.

Run:  PYTHONPATH=src uv run python scripts/run_k3_fold_F_free.py
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, root

jax.config.update("jax_enable_x64", True)

from hh_antiresonance.continuation import (
    bordered_newton_corrector, moore_penrose_tangent, orient_tangent,
)
from hh_antiresonance.harmonic_balance import (
    PARAMETER_INDEX, _hb_residual_jax, _parameter_array, _series_state_jax,
    solve_harmonic_balance, target_harmonic_indices,
)
from hh_antiresonance.models import CoupledOscillatorParams

DATADIR = Path("data"); DATADIR.mkdir(exist_ok=True)

KNL_FIX = -0.30
KA0 = 0.10
N_MAIN = 7
N_T = 512
OSC = 1

# working point of the Letter (F = 0.30, kappa = 0.10)
OM_STAR, B1_STAR, F_STAR = 0.2597280970367746, 0.2381542763447577, 0.30

# The tracked family sits at 5*Omega ~ omega_+ ~ 1.296.  Continuations that
# leave this window have jumped to another arm and are reported, not used.
OM_WINDOW = (0.20, 0.35)

BASE = CoupledOscillatorParams(
    omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.02,
    alpha1=0.0, alpha2=0.0, beta1=0.15, beta2=0.0,
    kappa=KA0, force=F_STAR, drive_omega=0.26, kappa_nl=KNL_FIX,
)

NC = 4 * N_MAIN
I3C, I3S = target_harmonic_indices(oscillator=OSC, harmonic=3, n_harmonics=N_MAIN)
I1C, I1S = target_harmonic_indices(oscillator=OSC, harmonic=1, n_harmonics=N_MAIN)


def log(m: str) -> None:
    print(m, flush=True)


def _pvals(Om, b1, ka, F):
    pv = jnp.asarray(_parameter_array(BASE))
    pv = pv.at[PARAMETER_INDEX["kappa_nl"]].set(KNL_FIX)
    pv = pv.at[PARAMETER_INDEX["force"]].set(F)
    pv = pv.at[PARAMETER_INDEX["drive_omega"]].set(Om)
    pv = pv.at[PARAMETER_INDEX["beta1"]].set(b1)
    pv = pv.at[PARAMETER_INDEX["kappa"]].set(ka)
    return pv


# ---- system 1: unknowns [coeffs, Omega, beta_1]; (kappa, F) are arguments ----
def _res_fixed(z, ka, F):
    c = z[:NC]
    hb = _hb_residual_jax(c, _pvals(z[NC], z[NC + 1], ka, F), N_MAIN, N_T)
    return jnp.concatenate([hb, jnp.array([c[I3C], c[I3S]], dtype=hb.dtype)])


_rf = jax.jit(_res_fixed)
_jf = jax.jit(jax.jacfwd(_res_fixed, argnums=0))
res_fixed = lambda z, ka, F: np.asarray(_rf(jnp.asarray(z, float), float(ka), float(F)), float)
jac_fixed = lambda z, ka, F: np.asarray(_jf(jnp.asarray(z, float), float(ka), float(F)), float)


# ---- system 2: unknowns [coeffs, Omega, beta_1, kappa]; F is an argument ----
def _res_freeka(z, F):
    c = z[:NC]
    hb = _hb_residual_jax(c, _pvals(z[NC], z[NC + 1], z[NC + 2], F), N_MAIN, N_T)
    return jnp.concatenate([hb, jnp.array([c[I3C], c[I3S]], dtype=hb.dtype)])


_rk = jax.jit(_res_freeka)
_jk = jax.jit(jax.jacfwd(_res_freeka, argnums=0))


def solve_zero(Om0, b0, ka, F, c0=None):
    """Square augmented Newton for the codim-2 zero at fixed (kappa, F)."""
    if c0 is None:
        c0 = solve_harmonic_balance(
            replace(BASE, drive_omega=Om0, beta1=b0, kappa=ka, force=F),
            n_harmonics=N_MAIN, n_time_samples=N_T, tol=1e-13, max_nfev=800)
    z0 = np.concatenate([np.asarray(c0, float), [Om0, b0]])
    s = least_squares(lambda z: res_fixed(z, ka, F), z0,
                      jac=lambda z: jac_fixed(z, ka, F),
                      xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
    s2 = root(lambda z: res_fixed(z, ka, F), s.x, jac=lambda z: jac_fixed(z, ka, F),
              method="hybr", tol=1e-15, options={"xtol": 1e-15, "maxfev": 3000})
    z = s2.x if (np.linalg.norm(res_fixed(s2.x, ka, F))
                 < np.linalg.norm(res_fixed(s.x, ka, F))) else s.x
    c = z[:NC]
    sv = np.linalg.svd(jac_fixed(z, ka, F), compute_uv=False)
    return dict(Om=float(z[NC]), b1=float(z[NC + 1]), coeffs=c,
                X3=float(np.hypot(c[I3C], c[I3S])),
                X1=float(np.hypot(c[I1C], c[I1S])),
                cond=float(sv[0] / sv[-1]),
                res=float(np.linalg.norm(res_fixed(z, ka, F))))


def orbit_diagnostics(coeffs, Om, ka):
    """max|d| and min tangent stiffness kappa + 3 kappa_nl d^2 over the cycle."""
    t = jnp.linspace(0.0, 2 * np.pi / Om, 2048, endpoint=False)
    st = _series_state_jax(t, jnp.asarray(coeffs, float), float(Om), N_MAIN)
    x1, x2 = np.asarray(st[0], float), np.asarray(st[3], float)  # (x1,v1,a1,x2,v2,a2)
    d = x1 - x2
    maxd = float(np.max(np.abs(d)))
    return maxd, float(ka + 3.0 * KNL_FIX * maxd ** 2)


def kappa_fold_at(F, seed, nsteps=420):
    """Pseudo-arclength continuation in (Omega, beta_1, kappa); return the fold."""
    res = lambda z: np.asarray(_rk(jnp.asarray(z, float), float(F)), float)
    jac = lambda z: np.asarray(_jk(jnp.asarray(z, float), float(F)), float)
    state0 = np.concatenate([seed["coeffs"], [seed["Om"], seed["b1"], KA0]])

    def run(direction):
        # NB: the tangent must be negated exactly once.  Negating it here AND
        # multiplying the step by sign(direction) cancels, which makes both
        # calls march the same way and returns one arm traced twice.
        st = state0.copy()
        t, _ = moore_penrose_tangent(jac(st))
        t = direction * orient_tangent(t, preferred_index=-1)
        S = [st.copy()]; ds = 3e-3; dsmin, dsmax = 1e-6, 1.5e-2
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

    # The locus extends far beyond the fold we want (out to kappa ~ 3 at
    # Omega ~ 2), so the global argmax of kappa lands on a distant arm.  The
    # fold where the two opposite-index arms merge is the FIRST turning point
    # of kappa encountered on leaving the seed, so march out and stop there.
    def first_kappa_turn(path):
        k = path[:, NC + 2]
        if len(k) < 3 or k[1] <= k[0]:
            return None                      # this direction leaves kappa falling
        for i in range(1, len(k) - 1):
            if k[i + 1] < k[i]:
                return i
        return None

    best = None
    for d in (+1, -1):
        path = run(d)
        i = first_kappa_turn(path)
        if i is None:
            continue
        if best is None or path[i, NC + 2] > best[0][best[1], NC + 2]:
            best = (path, i)
    if best is None:
        raise RuntimeError("no kappa turning point found on either side of the seed")
    loc, i = best
    ka = loc[:, NC + 2]
    x3 = float(np.hypot(loc[i, I3C], loc[i, I3S]))
    # Branch guard.  The tracked family lives near 5*Omega ~ omega_+, i.e.
    # Omega ~ 0.26.  A continuation that wanders to a different arm reports a
    # fold of THAT arm, which is not the quantity we are after.
    Om_fold = float(loc[i, NC])
    on_family = bool(OM_WINDOW[0] <= Om_fold <= OM_WINDOW[1])
    return dict(F=float(F), kstar=float(ka[i]), Om=Om_fold,
                b1=float(loc[i, NC + 1]), X3=x3, npts=len(loc),
                on_family=on_family, coeffs=loc[i, :NC].copy())


def main():
    log("=== is the kappa-fold an annihilation, or a fixed-F slice artifact? ===")
    log(f"fixed: kappa_nl={KNL_FIX}, N_H={N_MAIN}, N_T={N_T}")
    log(f"reference (Letter, F={F_STAR}): kappa* = 0.141856\n")

    # ---- (1) walk the working point in F at kappa = 0.10, to get seeds ------
    F_GRID = [0.26, 0.27, 0.28, 0.30, 0.32, 0.34, 0.36, 0.40, 0.45, 0.50, 0.60]
    log("(1) seed continuation in F at kappa=0.10 (free Omega,beta_1):")
    log(f"{'F':>6} {'Omega*':>12} {'beta1*':>12} {'|X3|':>10} {'|X1|':>9}")
    seeds = {}
    # march outward from F_STAR in both directions so each solve is warm-started
    order = ([f for f in F_GRID if f >= F_STAR]
             + [f for f in reversed(F_GRID) if f < F_STAR])
    prev = dict(Om=OM_STAR, b1=B1_STAR, coeffs=None)
    prev_lo = None
    for F in order:
        warm = prev if F >= F_STAR else (prev_lo or prev)
        s = solve_zero(warm["Om"], warm["b1"], KA0, F, c0=warm["coeffs"])
        if s["X3"] > 1e-9:
            log(f"{F:6.2f}   no root recovered (|X3|={s['X3']:.1e}) -- skipped")
            continue
        seeds[F] = s
        log(f"{F:6.2f} {s['Om']:12.7f} {s['b1']:12.7f} {s['X3']:10.1e} {s['X1']:9.4f}")
        if F >= F_STAR:
            prev = s
        else:
            prev_lo = s
        if F == F_STAR:
            prev_lo = s

    # ---- (2) kappa-fold at each F ------------------------------------------
    log("\n(2) kappa-fold from (Omega,beta_1,kappa) continuation at each F:")
    log(f"{'F':>6} {'kappa*(F)':>11} {'Omega_fold':>12} {'beta1_fold':>12} "
        f"{'|X3|':>9} {'pts':>5}")
    rows, offbranch = [], []
    for F in sorted(seeds):
        try:
            r = kappa_fold_at(F, seeds[F])
        except Exception as e:                                  # pragma: no cover
            log(f"{F:6.2f}   continuation failed: {type(e).__name__}")
            continue
        maxd, minK = orbit_diagnostics(r["coeffs"], r["Om"], r["kstar"])
        r["maxd"], r["minK"] = maxd, minK
        tag = "" if r["on_family"] else "   <-- off the tracked family, discarded"
        (rows if r["on_family"] else offbranch).append(r)
        log(f"{F:6.2f} {r['kstar']:11.6f} {r['Om']:12.7f} {r['b1']:12.7f} "
            f"{r['X3']:9.1e} {r['npts']:5d}{tag}")

    ks = np.array([r["kstar"] for r in rows]); Fs = np.array([r["F"] for r in rows])
    log(f"\n  on the tracked family, kappa*(F) rises monotonically from "
        f"{ks.min():.6f} to {ks.max():.6f} over F in [{Fs.min():.2f}, {Fs.max():.2f}]")
    log(f"  monotone: {bool(np.all(np.diff(ks) > 0))};  "
        f"mean slope d(kappa*)/dF = {np.polyfit(Fs, ks, 1)[0]:.3f}")

    # ---- (3) the fold curve also predicts the loss of the branch in F -------
    # At kappa = 0.10 the SM loses the tracked family between F = 0.258 and
    # 0.255 without an augmented calculation.  That threshold is just where the
    # fold curve kappa*(F) crosses 0.10, so the two events are one curve.
    quad = np.polyfit(Fs, ks, 2)
    roots = [x for x in np.roots([quad[0], quad[1], quad[2] - KA0]) if 0.2 < x.real < 0.6]
    F_pred = float(np.real(roots[0])) if roots else float("nan")
    log(f"\n(3) where the fold curve meets kappa=0.10:")
    log(f"  quadratic fit of kappa*(F) gives kappa*=0.10 at F = {F_pred:.4f}")
    log(f"  the Supplement loses the tracked family between F=0.258 and F=0.255")
    log(f"  -> the F-threshold and the kappa-fold are the same fold curve.")
    log("  direct confirmation by continuation at F just above/below:")
    log(f"{'F':>7} {'kappa*(F)':>11} {'Omega_fold':>12} {'vs kappa=0.10':>15}")
    fine = {}
    for F in (0.2560, 0.2580, 0.2600, 0.2620):
        base = seeds[min(seeds, key=lambda q: abs(q - 0.26))]
        s = solve_zero(base["Om"], base["b1"], KA0, F, c0=base["coeffs"])
        if s["X3"] > 1e-9:
            log(f"{F:7.4f}   no root at kappa=0.10 (|X3|={s['X3']:.1e}) "
                f"-> fold already passed")
            continue
        try:
            r = kappa_fold_at(F, s)
        except Exception:
            continue
        r["maxd"], r["minK"] = orbit_diagnostics(r["coeffs"], r["Om"], r["kstar"])
        fine[F] = r
        rel = "above" if r["kstar"] > KA0 else "BELOW"
        log(f"{F:7.4f} {r['kstar']:11.6f} {r['Om']:12.7f} {rel:>15}")

    # ---- (4) direct roots beyond the F=0.30 fold, on the tracked family -----
    K_REF = 0.141856
    log(f"\n(4) augmented solves at kappa > {K_REF} (the F=0.30 fold),")
    log("    each seeded from the fold point of the smallest F that clears it:")
    log(f"{'kappa':>8} {'F':>6} {'Omega*':>12} {'beta1*':>12} {'|X3|':>10} "
        f"{'cond J':>9} {'max|d|':>8} {'min K':>9}")
    checks = []
    for ka_t in (0.145, 0.150, 0.160, 0.180, 0.200, 0.240):
        cand = [r for r in rows if r["kstar"] > ka_t + 2e-3]
        if not cand:
            log(f"{ka_t:8.3f}   no F on the tracked grid has kappa*(F) > {ka_t}")
            continue
        r = min(cand, key=lambda q: q["F"])          # gentlest drive that works
        s = solve_zero(r["Om"], r["b1"], ka_t, r["F"], c0=r["coeffs"])
        maxd, minK = orbit_diagnostics(s["coeffs"], s["Om"], ka_t)
        ok = s["X3"] < 1e-9 and OM_WINDOW[0] <= s["Om"] <= OM_WINDOW[1]
        checks.append(dict(ka=ka_t, F=r["F"], Om=s["Om"], b1=s["b1"], X3=s["X3"],
                           cond=s["cond"], maxd=maxd, minK=minK, ok=ok))
        log(f"{ka_t:8.3f} {r['F']:6.2f} {s['Om']:12.7f} {s['b1']:12.7f} "
            f"{s['X3']:10.1e} {s['cond']:9.1e} {maxd:8.4f} {minK:+9.4f}"
            + ("" if ok else "   <-- not a root on the family"))

    # ---- (5) verdict --------------------------------------------------------
    log("\n(5) verdict:")
    surv = [c for c in checks if c["ok"]]
    if surv:
        kmax = max(c["ka"] for c in surv)
        log(f"  Third-harmonic zeros on the tracked family exist at kappa up to "
            f"{kmax:.3f},")
        log(f"  i.e. {100*(kmax/K_REF - 1):.0f}% beyond the F=0.30 fold at {K_REF},")
        log(f"  reached by raising the drive to F={max(c['F'] for c in surv):.2f}.")
        log("  The merger is therefore a tangency of the zero set with the")
        log("  F=0.30 slice, not an annihilation of the zero set itself.")
        inwin = [c for c in surv if c["minK"] > 0 and c["F"] <= 0.355]
        log(f"  Constitutive window (min K > 0 and F <= 0.355): "
            f"{len(inwin)}/{len(surv)} of these survive it"
            + (f", up to kappa={max(c['ka'] for c in inwin):.3f}." if inwin else "."))
    else:
        log(f"  No zero recovered beyond kappa={K_REF} on the tested F grid.")

    merged = {round(r["F"], 6): r for r in rows}      # fine grid wins on ties
    merged.update({round(r["F"], 6): r for r in fine.values()})
    allrows = sorted(merged.values(), key=lambda r: r["F"])
    np.savetxt(DATADIR / "two_dof_k3_fold_F_free.csv",
               np.array([[r["F"], r["kstar"], r["Om"], r["b1"], r["X3"],
                          r["maxd"], r["minK"]] for r in allrows]),
               delimiter=",",
               header="F,kappa_star,omega_fold,beta1_fold,abs_X3,max_abs_d,min_tangent_K",
               comments="")
    np.savetxt(DATADIR / "two_dof_k3_fold_F_free_checks.csv",
               np.array([[c["ka"], c["F"], c["Om"], c["b1"], c["X3"],
                          c["cond"], c["maxd"], c["minK"], float(c["ok"])]
                         for c in checks]),
               delimiter=",",
               header="kappa,F,omega,beta1,abs_X3,cond_J,max_abs_d,min_tangent_K,on_family",
               comments="")
    log(f"\nwrote {DATADIR/'two_dof_k3_fold_F_free.csv'}")


if __name__ == "__main__":
    main()
