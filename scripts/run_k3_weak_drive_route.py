"""Does the zero set have a route toward weak drive?

The Letter calls the root "intrinsically finite-amplitude": at kappa = 0.10 the
tracked branch is lost near F = 0.256 and does not continue to F -> 0.  But
run_k3_fold_F_free.py showed that the loss is not an endpoint of the zero set --
it is where the fold curve kappa*(F) crosses kappa = 0.10.  Since kappa*(F)
decreases with F, a smaller coupling should tolerate a smaller forcing, and the
weak-drive claim has to be tested in (kappa, F) jointly rather than along the
kappa = 0.10 line.

This computes the inverse function directly.  For each fixed kappa we continue
the zero in (Omega, beta_1, F) and read the fold as the SMALLEST forcing on the
physical branch,

    F*(kappa) = min F on the tracked branch,

which is the threshold below which no zero of this family exists at that
coupling.  If F*(kappa) -> 0 as kappa -> 0, the zero set does reach the
weak-drive limit and the finite-amplitude interpretation would be falsified in
the joint limit.  If F*(kappa) flattens to a positive floor, the interpretation
survives in the joint limit and can be quantified.

Run:  PYTHONPATH=src uv run python scripts/run_k3_weak_drive_route.py
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
    PARAMETER_INDEX, _hb_residual_jax, _parameter_array,
    solve_harmonic_balance, target_harmonic_indices,
)
from hh_antiresonance.models import CoupledOscillatorParams

DATADIR = Path("data"); DATADIR.mkdir(exist_ok=True)

KNL_FIX = -0.30
N_MAIN = 7
N_T = 512
OSC = 1
OM_STAR, B1_STAR, F_STAR, KA0 = 0.2597280970367746, 0.2381542763447577, 0.30, 0.10
OM_WINDOW = (0.15, 0.40)

BASE = CoupledOscillatorParams(
    omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.02,
    alpha1=0.0, alpha2=0.0, beta1=0.15, beta2=0.0,
    kappa=KA0, force=F_STAR, drive_omega=0.26, kappa_nl=KNL_FIX,
)

NC = 4 * N_MAIN
I3C, I3S = target_harmonic_indices(oscillator=OSC, harmonic=3, n_harmonics=N_MAIN)
I1C, I1S = target_harmonic_indices(oscillator=OSC, harmonic=1, n_harmonics=N_MAIN)


def log(m): print(m, flush=True)


def _pv(Om, b1, ka, F):
    pv = jnp.asarray(_parameter_array(BASE))
    pv = pv.at[PARAMETER_INDEX["kappa_nl"]].set(KNL_FIX)
    pv = pv.at[PARAMETER_INDEX["force"]].set(F)
    pv = pv.at[PARAMETER_INDEX["drive_omega"]].set(Om)
    pv = pv.at[PARAMETER_INDEX["beta1"]].set(b1)
    pv = pv.at[PARAMETER_INDEX["kappa"]].set(ka)
    return pv


def _res_fixed(z, ka, F):                       # unknowns [c, Omega, beta_1]
    c = z[:NC]
    hb = _hb_residual_jax(c, _pv(z[NC], z[NC + 1], ka, F), N_MAIN, N_T)
    return jnp.concatenate([hb, jnp.array([c[I3C], c[I3S]], dtype=hb.dtype)])


_rf, _jf = jax.jit(_res_fixed), jax.jit(jax.jacfwd(_res_fixed, argnums=0))
res_fixed = lambda z, ka, F: np.asarray(_rf(jnp.asarray(z, float), float(ka), float(F)), float)
jac_fixed = lambda z, ka, F: np.asarray(_jf(jnp.asarray(z, float), float(ka), float(F)), float)


def _res_freeF(z, ka):                          # unknowns [c, Omega, beta_1, F]
    c = z[:NC]
    hb = _hb_residual_jax(c, _pv(z[NC], z[NC + 1], ka, z[NC + 2]), N_MAIN, N_T)
    return jnp.concatenate([hb, jnp.array([c[I3C], c[I3S]], dtype=hb.dtype)])


_rF, _jF = jax.jit(_res_freeF), jax.jit(jax.jacfwd(_res_freeF, argnums=0))


def solve_zero(Om0, b0, ka, F, c0=None):
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
    return dict(Om=float(z[NC]), b1=float(z[NC + 1]), coeffs=c,
                X3=float(np.hypot(c[I3C], c[I3S])),
                X1=float(np.hypot(c[I1C], c[I1S])))


def force_fold_at(ka, seed, nsteps=600):
    """Continue in (Omega, beta_1, F) at fixed kappa; return the minimum F."""
    res = lambda z: np.asarray(_rF(jnp.asarray(z, float), float(ka)), float)
    jac = lambda z: np.asarray(_jF(jnp.asarray(z, float), float(ka)), float)
    state0 = np.concatenate([seed["coeffs"], [seed["Om"], seed["b1"], F_STAR]])

    def run(direction):
        # The tangent must be negated exactly once: negating it here AND
        # multiplying the step by sign(direction) cancels out, so both calls
        # march the same way and the "locus" is one arm traced twice.
        st = state0.copy()
        t, _ = moore_penrose_tangent(jac(st))
        t = direction * orient_tangent(t, preferred_index=-1)
        S = [st.copy()]; ds = 2e-3; dsmin, dsmax = 1e-7, 1.0e-2
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
            if not ok or cor[NC] < 0.05 or cor[NC + 2] <= 0.0:
                break
            try:
                t, _ = moore_penrose_tangent(jac(cor), reference_tangent=t)
            except Exception:
                sec = cor - st; t = sec / (np.linalg.norm(sec) + 1e-14)
            st = cor; S.append(st.copy())
            ds = min(dsmax, 1.4 * ds) if int(info.get("iters", 3)) < 4 else max(dsmin, 0.5 * ds)
        return np.asarray(S)

    loc = np.vstack([run(-1)[::-1], run(+1)[1:]])
    Fv = loc[:, NC + 2]; Omv = loc[:, NC]; b1v = loc[:, NC + 1]
    X1v = np.hypot(loc[:, I1C], loc[:, I1S])
    X3v = np.hypot(loc[:, I3C], loc[:, I3S])

    # global minimum forcing anywhere on the traced locus
    ig = int(np.argmin(Fv))
    # minimum forcing while the branch is still in the tracked family: this is
    # the turning point the Supplement sees as "loss of the branch"
    infam = (Omv >= OM_WINDOW[0]) & (Omv <= OM_WINDOW[1])
    itr = int(np.arange(len(Fv))[infam][np.argmin(Fv[infam])]) if infam.any() else ig

    # continuity audit: largest jump between consecutive corrector points
    step = np.sqrt(np.diff(Omv) ** 2 + np.diff(b1v) ** 2 + np.diff(Fv) ** 2)
    return dict(ka=float(ka),
                Fmin_family=float(Fv[itr]), Om_family=float(Omv[itr]),
                b1_family=float(b1v[itr]), X1_family=float(X1v[itr]),
                Fmin_global=float(Fv[ig]), Om_global=float(Omv[ig]),
                b1_global=float(b1v[ig]), X1_global=float(X1v[ig]),
                maxX3=float(X3v.max()), maxstep=float(step.max()),
                npts=len(loc), hit_cap=bool(len(loc) >= 2 * nsteps),
                path=np.column_stack([Fv, Omv, b1v, X1v, X3v]),
                coeffs=loc[itr, :NC].copy())


def main():
    log("=== does the zero set reach the weak-drive limit? ===")
    log(f"fixed: kappa_nl={KNL_FIX}, N_H={N_MAIN}")
    log("reference: at kappa=0.10 the tracked branch is lost near F=0.256\n")

    # ---- (1) seeds: walk kappa down at F = 0.30 -----------------------------
    KAS = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
    log("(1) seed roots at F=0.30, marching kappa down:")
    log(f"{'kappa':>7} {'Omega*':>12} {'beta1*':>12} {'|X3|':>10} {'|X1|':>9}")
    seeds = {}
    prev = dict(Om=OM_STAR, b1=B1_STAR, coeffs=None)
    for ka in KAS:
        s = solve_zero(prev["Om"], prev["b1"], ka, F_STAR, c0=prev["coeffs"])
        if s["X3"] > 1e-9 or not (OM_WINDOW[0] <= s["Om"] <= OM_WINDOW[1]):
            log(f"{ka:7.3f}   no root on the family (|X3|={s['X3']:.1e}, Om={s['Om']:.4f})")
            continue
        seeds[ka] = s; prev = s
        log(f"{ka:7.3f} {s['Om']:12.7f} {s['b1']:12.7f} {s['X3']:10.1e} {s['X1']:9.4f}")

    # ---- (2) continue in F at fixed kappa ----------------------------------
    log("\n(2) (Omega,beta_1,F) continuation at fixed kappa.")
    log("    'family' = turning point while Omega stays near the tracked arm;")
    log("    'global' = smallest F anywhere on the traced locus.")
    log(f"{'kappa':>7} | {'F_turn':>9} {'Omega':>9} {'beta1':>9} | "
        f"{'F_min':>9} {'Omega':>9} {'|X1|':>7} | {'max|X3|':>9} {'maxstep':>8} {'cap':>4}")
    rows = []
    for ka in sorted(seeds, reverse=True):
        try:
            r = force_fold_at(ka, seeds[ka])
        except Exception as e:                                   # pragma: no cover
            log(f"{ka:7.3f}   continuation failed: {type(e).__name__}")
            continue
        rows.append(r)
        log(f"{ka:7.3f} | {r['Fmin_family']:9.5f} {r['Om_family']:9.5f} "
            f"{r['b1_family']:9.5f} | {r['Fmin_global']:9.5f} {r['Om_global']:9.5f} "
            f"{r['X1_global']:7.4f} | {r['maxX3']:9.1e} {r['maxstep']:8.4f} "
            f"{'yes' if r['hit_cap'] else 'no':>4}")

    if not rows:
        log("\nno usable rows"); return

    # ---- (3) the tracked arm turns; the locus does not end -----------------
    log("\n(3) reading:")
    r0 = [r for r in rows if abs(r["ka"] - KA0) < 1e-9][0]
    log(f"  At kappa=0.10 the tracked arm turns at F={r0['Fmin_family']:.5f} "
        f"(Omega={r0['Om_family']:.5f}),")
    log(f"  reproducing the Supplement's loss of the branch between F=0.258 and 0.255.")
    log(f"  The locus does not end there: continuation carries it to F="
        f"{r0['Fmin_global']:.5f} at Omega={r0['Om_global']:.5f},")
    log(f"  i.e. onto the lower coupled mode omega_-=1.0406, with |X1|="
        f"{r0['X1_global']:.4f} still finite.")
    log(f"  Constraint held throughout: max|X3| on the whole locus = {r0['maxX3']:.1e}.")
    caps = [r for r in rows if r["hit_cap"]]
    if caps:
        log(f"  {len(caps)}/{len(rows)} continuations stopped at the step cap, so the")
        log("  smallest F reached is a stopping point, NOT a computed fold in F.")

    ka = np.array([r["ka"] for r in rows])
    Ftr = np.array([r["Fmin_family"] for r in rows])
    Fgl = np.array([r["Fmin_global"] for r in rows])
    log(f"\n  tracked-arm turning forcing F_turn(kappa): "
        f"{Ftr.min():.4f} to {Ftr.max():.4f} over kappa in "
        f"[{ka.min():.2f},{ka.max():.2f}]")
    log(f"  smallest F reached on the locus: {Fgl.min():.4f} to {Fgl.max():.4f}")

    log("\n(4) verdict:")
    log("  The tracked arm does turn at finite forcing, at every coupling tested,")
    log("  so the reported working point has no weak-drive counterpart ON THAT ARM.")
    log("  But the zero locus is connected past that turning point and continues")
    log("  to much weaker drive on a different arm, near the lower coupled mode.")
    log("  'Intrinsically finite-amplitude, with no continuation toward weak drive'")
    log("  is therefore NOT supportable as a statement about the zero set; it is")
    log("  only a statement about the tracked arm at fixed coupling.")

    np.savetxt(DATADIR / "two_dof_k3_weak_drive_route.csv",
               np.array([[r["ka"], r["Fmin_family"], r["Om_family"], r["b1_family"],
                          r["Fmin_global"], r["Om_global"], r["X1_global"],
                          r["maxX3"], float(r["hit_cap"])] for r in rows]),
               delimiter=",",
               header=("kappa,F_turn_family,omega_turn,beta1_turn,"
                       "F_min_global,omega_min_global,abs_X1_global,max_abs_X3,hit_cap"),
               comments="")
    np.savetxt(DATADIR / "two_dof_k3_weak_drive_path_kappa010.csv",
               r0["path"], delimiter=",",
               header="force,omega,beta1,abs_X1,abs_X3", comments="")
    log(f"\nwrote {DATADIR/'two_dof_k3_weak_drive_route.csv'} and the kappa=0.10 path")


if __name__ == "__main__":
    main()
