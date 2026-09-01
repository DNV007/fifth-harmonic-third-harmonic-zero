"""Non-degenerate k=3 third-harmonic exact zero and its genuine saddle-node fold.

This produces the non-degenerate codim-2 third-harmonic exact zero of Sec.~S19.
An earlier demonstrator found the codim-2 third-harmonic exact zero with kappa as
the unfolding parameter at low forcing (F=0.05); there the seed sat on a kappa-fold
of itself,
so the augmented Jacobian was rank-deficient and |X3| stalled at ~6.17e-8 -- a
*degenerate* proof-of-existence whose fold was a single curve folding back on
itself (cusp-like), not the generic two-arm saddle-node of Eq.~(8).

Here we obtain a NON-degenerate working point by
  (a) switching the unfolding parameter from kappa to beta_1 (the cubic stiffness
      has no trivial mechanism to null X3, unlike the Omega->0 / F->0 sheets), and
  (b) raising the forcing to F=0.30, closer to the 3:1 region where the third
      harmonic is amplified and its sign change through zero is sharp.

Pipeline (single 2-DOF coupled-Duffing system):

  (1) Codim-2 exact zero.  Free (Omega, beta_1) at fixed (kappa, F, kappa_nl);
      solve the square augmented system [HB ; X3^cos ; X3^sin] = 0.  Repeat at
      N_H = 5,6,7,8 to show truncation convergence and report the augmented-
      Jacobian condition number (full rank == non-degenerate).
  (2) DOP853 cross-check.  Direct steady-state time integration at the converged
      seed reproduces A1 and drives A3 to the integrator floor: the exact zero is
      genuine dynamics, not an HB-closure artifact.
  (3) Two-arm saddle-node.  For kappa in [0.10, kappa*], track BOTH branches of
      the exact zero; they merge at kappa* with canonical sqrt(kappa* - kappa)
      separation.  |X3| stays at the machine floor on both arms throughout -- the
      constraint is re-enforced, not carried.
  (4) Fold locus.  Free (Omega, beta_1, kappa) pseudo-arclength continuation at
      tight corrector tolerance, restricted to the physical branch (Omega>0.05);
      the kappa-fold is the observable codim-1 landmark of THIS slice under THIS
      projection.  It is not an invariant of the zero set: freeze different
      controls, or project on a different one, and the fold moves.  Only its
      existence and the sqrt branch-separation law are choice-independent.
  (5) Persist data (data/two_dof_k3_nondeg_*.csv) + figure
      (figures/k3_nondeg_exact_zero_fold.{pdf,png}).

Run:  PYTHONPATH=src uv run python scripts/run_k3_nondegenerate_exact_zero.py
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import curve_fit, least_squares, root
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from hh_antiresonance import figstyle as fs

fs.use()

from hh_antiresonance.continuation import (
    bordered_newton_corrector, detect_parameter_folds,
    moore_penrose_tangent, orient_tangent,
)
from hh_antiresonance.harmonic_balance import (
    PARAMETER_INDEX, _hb_residual_jax, _parameter_array,
    solve_harmonic_balance, target_harmonic_indices,
)
from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.time_integration import integrate_steady_state

DATADIR = Path("data"); FIGDIR = Path("figures")
DATADIR.mkdir(exist_ok=True); FIGDIR.mkdir(exist_ok=True)

# ---- working point (non-degenerate) -------------------------------------
F_FIX = 0.30       # higher forcing: sharp third-harmonic sign change
KNL_FIX = -0.30    # fixed reaction nonlinearity
KA0 = 0.10         # base coupling (seed side of the saddle-node)
N_MAIN = 7
N_T = 512
OSC, HARM = 1, 3

BASE = CoupledOscillatorParams(
    omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.02,
    alpha1=0.0, alpha2=0.0, beta1=0.15, beta2=0.0,
    kappa=KA0, force=F_FIX, drive_omega=0.26, kappa_nl=KNL_FIX,
)


def log(m: str) -> None:
    print(m, flush=True)


# -------------------- augmented defining system ---------------------------
# Square fixed-kappa system: unknowns z=[coeffs(4N), Omega, beta_1]; kappa is a
# runtime argument (so the two-arm sweep genuinely varies the coupling).
# Continuation system: unknowns z=[coeffs(4N), Omega, beta_1, kappa].

_FIXED_CACHE: dict[int, tuple] = {}


def _build_fixed(N_H):
    NC = 4 * N_H
    i3c, i3s = target_harmonic_indices(oscillator=OSC, harmonic=3, n_harmonics=N_H)
    i1c, i1s = target_harmonic_indices(oscillator=OSC, harmonic=1, n_harmonics=N_H)

    def residual_jax(z, ka):
        c = z[:NC]
        Om = z[NC]; b1 = z[NC + 1]
        pv = jnp.asarray(_parameter_array(BASE))
        pv = pv.at[PARAMETER_INDEX["force"]].set(F_FIX)
        pv = pv.at[PARAMETER_INDEX["kappa_nl"]].set(KNL_FIX)
        pv = pv.at[PARAMETER_INDEX["drive_omega"]].set(Om)
        pv = pv.at[PARAMETER_INDEX["beta1"]].set(b1)
        pv = pv.at[PARAMETER_INDEX["kappa"]].set(ka)
        hb = _hb_residual_jax(c, pv, N_H, N_T)
        return jnp.concatenate([hb, jnp.array([c[i3c], c[i3s]], dtype=hb.dtype)])

    res_jit = jax.jit(residual_jax)
    jac_jit = jax.jit(jax.jacfwd(residual_jax, argnums=0))
    res = lambda z, ka: np.asarray(res_jit(jnp.asarray(z, float), float(ka)), float)
    jac = lambda z, ka: np.asarray(jac_jit(jnp.asarray(z, float), float(ka)), float)
    return res, jac, NC, (i3c, i3s, i1c, i1s)


def get_fixed(N_H):
    if N_H not in _FIXED_CACHE:
        _FIXED_CACHE[N_H] = _build_fixed(N_H)
    return _FIXED_CACHE[N_H]


def make_system(N_H, free_kappa=True):
    """Free-kappa continuation system, unknowns [coeffs, Omega, beta_1, kappa]."""
    NC = 4 * N_H
    i3c, i3s = target_harmonic_indices(oscillator=OSC, harmonic=3, n_harmonics=N_H)
    i1c, i1s = target_harmonic_indices(oscillator=OSC, harmonic=1, n_harmonics=N_H)

    def residual_jax(z):
        c = z[:NC]
        Om = z[NC]; b1 = z[NC + 1]; ka = z[NC + 2]
        pv = jnp.asarray(_parameter_array(BASE))
        pv = pv.at[PARAMETER_INDEX["force"]].set(F_FIX)
        pv = pv.at[PARAMETER_INDEX["kappa_nl"]].set(KNL_FIX)
        pv = pv.at[PARAMETER_INDEX["drive_omega"]].set(Om)
        pv = pv.at[PARAMETER_INDEX["beta1"]].set(b1)
        pv = pv.at[PARAMETER_INDEX["kappa"]].set(ka)
        hb = _hb_residual_jax(c, pv, N_H, N_T)
        return jnp.concatenate([hb, jnp.array([c[i3c], c[i3s]], dtype=hb.dtype)])

    res_jit = jax.jit(residual_jax)
    jac_jit = jax.jit(jax.jacfwd(residual_jax))
    res = lambda z: np.asarray(res_jit(jnp.asarray(z, float)), float)
    jac = lambda z: np.asarray(jac_jit(jnp.asarray(z, float)), float)
    return res, jac, NC, (i3c, i3s, i1c, i1s)


def solve_zero(N_H, Om0, b0, ka=KA0):
    """Square augmented Newton for the codim-2 zero at fixed kappa=ka."""
    res, jac, NC, (i3c, i3s, i1c, i1s) = get_fixed(N_H)
    c0 = solve_harmonic_balance(replace(BASE, drive_omega=Om0, beta1=b0, kappa=ka),
                                n_harmonics=N_H, n_time_samples=N_T, tol=1e-13, max_nfev=800)
    z0 = np.concatenate([np.asarray(c0, float), [Om0, b0]])
    s = least_squares(lambda z: res(z, ka), z0, jac=lambda z: jac(z, ka),
                      xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
    s2 = root(lambda z: res(z, ka), s.x, jac=lambda z: jac(z, ka),
              method="hybr", tol=1e-15, options={"xtol": 1e-15, "maxfev": 3000})
    z = s2.x if np.linalg.norm(res(s2.x, ka)) < np.linalg.norm(res(s.x, ka)) else s.x
    c = z[:NC]
    sv = np.linalg.svd(jac(z, ka), compute_uv=False)
    return dict(N=N_H, Om=float(z[NC]), b1=float(z[NC + 1]),
                X3=float(np.hypot(c[i3c], c[i3s])), X1=float(np.hypot(c[i1c], c[i1s])),
                cond=float(sv[0] / sv[-1]), res=float(np.linalg.norm(res(z, ka))),
                coeffs=c, i3=(i3c, i3s), i1=(i1c, i1s))


# =========================================================================
def main():
    log("=== non-degenerate k=3 exact zero (beta_1-unfolding, F=0.30) ===")
    log(f"fixed: kappa={KA0}, F={F_FIX}, kappa_nl={KNL_FIX}; N_main={N_MAIN}, N_T={N_T}")

    # ---------- (1) codim-2 zero, truncation convergence ----------
    log("\n(1) codim-2 exact zero, free (Omega,beta_1), vs truncation N:")
    log(f"{'N':>3} {'Omega*':>12} {'beta1*':>12} {'|X3|':>11} {'|X1|':>10} {'cond(J)':>10}")
    conv_rows = []
    Om_c, b_c = 0.2597, 0.2382
    conv_seed = None
    for N in (5, 6, 7, 8):
        d = solve_zero(N, Om_c, b_c)
        Om_c, b_c = d["Om"], d["b1"]
        log(f"{d['N']:3d} {d['Om']:12.7f} {d['b1']:12.7f} {d['X3']:11.2e} {d['X1']:10.6f} {d['cond']:10.1e}")
        conv_rows.append((N, d["Om"], d["b1"], d["X3"], d["X1"], d["cond"]))
        if N == N_MAIN:
            conv_seed = d
    Om_star, b_star = conv_seed["Om"], conv_seed["b1"]

    # ---------- (2) DOP853 cross-check at the converged seed ----------
    log(f"\n(2) DOP853 steady-state at Omega*={Om_star:.6f}, beta1*={b_star:.6f}:")
    p = replace(BASE, drive_omega=Om_star, beta1=b_star, kappa=KA0, force=F_FIX, kappa_nl=KNL_FIX)
    rep = integrate_steady_state(p, max_harmonic=5, n_transient_periods=400,
                                 n_steady_periods=64, samples_per_period=512,
                                 ode_method="DOP853", rtol=1e-12, atol=1e-14)
    A1_dop, A3_dop = float(rep.amplitudes_osc1[0]), float(rep.amplitudes_osc1[2])
    log(f"  HB(N={N_MAIN}): A1={conv_seed['X1']:.6e}  A3={conv_seed['X3']:.3e}")
    log(f"  DOP853:    A1={A1_dop:.6e}  A3={A3_dop:.3e}  periodicity_resid={rep.periodicity_residual:.2e}")

    # ---------- (3) two-arm saddle-node ----------
    log("\n(3) two-arm saddle-node: both branches of the exact zero vs kappa")
    log(f"{'kappa':>7} | {'OmA':>8} {'beA':>8} {'|X3|A':>9} | {'OmB':>8} {'beB':>8} {'|X3|B':>9} | {'dbeta':>8}")
    # kappa grid stops short of the fold: within ~1e-3 of kappa* the two arms are
    # closer than the Newton basin and a warm-started solve on one arm jumps to
    # the other.  We fit the sqrt law from the clean, well-separated points and
    # cross-check kappa* against the locus fold (part 4).
    kgrid = np.array([0.100, 0.110, 0.120, 0.126, 0.130, 0.134, 0.137, 0.139, 0.140, 0.141])
    OmA, bA = 0.25973, 0.23815   # branch A seed (== the N=7 converged zero)
    OmB, bB = 0.25152, 0.24022   # branch B seed (second arm at kappa=0.10)
    sn_rows = []
    for ka in kgrid:
        dA = solve_zero(N_MAIN, OmA, bA, ka=ka); OmA, bA = dA["Om"], dA["b1"]
        dB = solve_zero(N_MAIN, OmB, bB, ka=ka); OmB, bB = dB["Om"], dB["b1"]
        sn_rows.append((ka, dA["Om"], dA["b1"], dA["X3"], dA["X1"],
                        dB["Om"], dB["b1"], dB["X3"], dB["X1"]))
        log(f"{ka:7.4f} | {dA['Om']:8.5f} {dA['b1']:8.5f} {dA['X3']:9.1e} | "
            f"{dB['Om']:8.5f} {dB['b1']:8.5f} {dB['X3']:9.1e} | {abs(dA['Om']-dB['Om']):8.5f}")
    sn = np.array(sn_rows)
    dsep = np.abs(sn[:, 1] - sn[:, 5])                # |Omega_A - Omega_B| (cleaner arm)
    collapsed = dsep < 1e-4                            # arms merged into solver tol
    mask = (~collapsed) & (dsep > 5e-4)
    max_x3_sn = float(max(sn[:, 3].max(), sn[:, 7].max()))
    # A single wide-window fit that *assumes* p=1/2 (dsep^2 linear in kappa) lands on
    # kappa*=0.1396, low by 1.6% against the directly solved apex.  It is computed here
    # only as the biased reference that the shrinking-window analysis in (4b) replaces;
    # it is neither plotted nor quoted.
    pfit_wide = np.polyfit(sn[mask, 0], dsep[mask] ** 2, 1)
    kstar_wide = float(-pfit_wide[1] / pfit_wide[0])
    log(f"  wide-window fit, p forced to 1/2 (biased, not used): kappa* = {kstar_wide:.5f}")
    log(f"  max |X3| over both arms, all kappa = {max_x3_sn:.2e}  (machine floor; re-enforced)")

    # ---------- (4) fold locus (free Omega,beta_1,kappa) ----------
    log(f"\n(4) fold-locus continuation at N={N_MAIN}, tight tol, physical branch (Omega>0.05):")
    res, jac, NC, (i3c, i3s, i1c, i1s) = make_system(N_MAIN, free_kappa=True)
    c7 = solve_harmonic_balance(replace(BASE, drive_omega=Om_star, beta1=b_star, kappa=KA0),
                                n_harmonics=N_MAIN, n_time_samples=N_T, tol=1e-13)
    state0 = np.concatenate([np.asarray(c7, float), [Om_star, b_star, KA0]])
    X3 = lambda s: float(np.hypot(s[i3c], s[i3s]))
    X1 = lambda s: float(np.hypot(s[i1c], s[i1s]))

    def run(direction, nsteps=260):
        # The tangent must be negated exactly once.  Negating it here AND
        # multiplying the step by sign(direction) below cancels out, so both
        # calls march the same way and the stitched "locus" is one arm traced
        # twice (visible as a locus mirror-symmetric about the seed index).
        st = state0.copy()
        t, _ = moore_penrose_tangent(jac(st))
        t = direction * orient_tangent(t, preferred_index=-1)
        S = [st.copy()]; ds = 3e-3; dsmin, dsmax = 1e-6, 1.5e-2
        for _ in range(nsteps):
            ok = False
            for _try in range(40):
                pred = st + ds * t
                try:
                    cor, info = bordered_newton_corrector(res, pred, t,
                                                          jac_func=jac, tol=1e-10, max_iter=40)
                except Exception:
                    info = {"converged": False}; cor = pred
                if info["converged"]:
                    ok = True; break
                ds *= 0.5
                if ds < dsmin:
                    break
            if not ok or cor[NC] < 0.05:   # stop at unphysical Omega
                break
            try:
                t, _ = moore_penrose_tangent(jac(cor), reference_tangent=t)
            except Exception:
                sec = cor - st; t = sec / (np.linalg.norm(sec) + 1e-14)
            st = cor; S.append(st.copy())
            ds = min(dsmax, 1.4 * ds) if int(info.get("iters", 3)) < 4 else max(dsmin, 0.5 * ds)
        return np.asarray(S)

    sp = run(+1); sm = run(-1)
    loc = np.vstack([sm[::-1], sp[1:]])
    om = loc[:, NC]; be = loc[:, NC + 1]; ka = loc[:, NC + 2]
    x3 = np.array([X3(s) for s in loc]); x1 = np.array([X1(s) for s in loc])
    fold_idx = detect_parameter_folds(ka)

    # The apex is the FIRST turning point of kappa on leaving the seed, not the
    # global argmax: once both directions are traced the locus runs far past
    # this fold (out to large kappa at high Omega), and argmax lands there.
    i_seed = len(sm) - 1                      # index of state0 in the stitched locus
    def _first_turn(idx_range):
        prev = ka[i_seed]; last = i_seed
        for i in idx_range:
            if ka[i] < prev:
                return last
            prev = ka[i]; last = i
        return None
    cands = [j for j in (_first_turn(range(i_seed + 1, len(ka))),
                         _first_turn(range(i_seed - 1, -1, -1))) if j is not None]
    imax = max(cands, key=lambda j: ka[j]) if cands else int(np.argmax(ka))
    log(f"  physical locus: {len(om)} pts; Omega=[{om.min():.4f},{om.max():.4f}] "
        f"beta1=[{be.min():.4f},{be.max():.4f}] kappa=[{ka.min():.4f},{ka.max():.4f}]")
    log(f"  max|X3| on locus = {x3.max():.2e}  median={np.median(x3):.2e}  (re-enforced)")
    log(f"  kappa-fold (max kappa on physical branch): kappa*={ka[imax]:.5f} "
        f"Omega={om[imax]:.5f} beta1={be[imax]:.5f} |X3|={x3[imax]:.2e}")

    # ---------- (4b) square-root law by shrinking-window convergence ----------
    # Two fits per window, and the manuscript quotes the SECOND one.
    #
    #   free      (kappa*, A, p) all free, least squares in the original units.
    #             A consistency check: p -> 1/2 and the free apex -> the directly
    #             solved kappa* as the window shrinks, and a wide window is
    #             biased high in p.
    #   fixed     kappa* pinned at the augmented turning-point value, so only
    #             (A, p) are fitted, as a least-squares regression of
    #             log|Omega_A - Omega_B| on log(kappa* - kappa).  The log form
    #             gives each arm equal weight in the exponent instead of letting
    #             the widely separated arms dominate; the rms is reported back in
    #             the original units so the two columns are comparable.
    #
    # The apex should not be fitted twice, so the fixed variant is the reported
    # one: it is the source of the p = 0.511 (delta = 0.005) and p = 0.547
    # (delta = 0.040) values quoted in the Letter's fold-figure caption and in
    # Sec. S6 of the Supplemental Material.  Fitting the fixed variant in the
    # original units instead would return 0.513 and 0.583, so the estimator is
    # part of the reported number and is recorded here rather than left implicit.
    kfold = float(ka[imax])          # authoritative apex from the locus continuation
    _law = lambda k, ks, A, p: A * np.maximum(ks - k, 1e-12) ** p

    def shrink_fit(delta):
        m = mask & (kfold - sn[:, 0] <= delta)
        if int(m.sum()) < 3:
            return None
        popt, _ = curve_fit(_law, sn[m, 0], dsep[m], p0=[kfold, 0.2, 0.5], maxfev=400_000)
        rms = float(np.sqrt(np.mean((_law(sn[m, 0], *popt) - dsep[m]) ** 2)))

        # kappa* pinned, fitted in log-log; rms converted back to original units
        gap = kfold - sn[m, 0]
        slope, intercept = np.polyfit(np.log(gap), np.log(dsep[m]), 1)
        amp_fix = float(np.exp(intercept))
        pred = amp_fix * gap ** slope
        rms_fix = float(np.sqrt(np.mean((pred - dsep[m]) ** 2)))

        return dict(delta=float(delta), npts=int(m.sum()), kstar=float(popt[0]),
                    amp=float(popt[1]), p=float(popt[2]), rms=rms,
                    amp_fixed=amp_fix, p_fixed=float(slope), rms_fixed=rms_fix)

    windows = [w for w in (shrink_fit(d) for d in (0.040, 0.020, 0.010, 0.005)) if w]
    log("\n(4b) shrinking-window fits  |Omega_A-Omega_B| = A (kappa*-kappa)^p:")
    log(f"{'delta':>7} {'npts':>5} {'kappa*_fit':>11} {'p_free':>8} {'RMS':>10}"
        f" {'p_fixed':>9} {'RMS_fixed':>11}")
    for w in windows:
        log(f"{w['delta']:7.3f} {w['npts']:5d} {w['kstar']:11.5f} "
            f"{w['p']:8.3f} {w['rms']:10.1e} {w['p_fixed']:9.3f} {w['rms_fixed']:11.1e}")
    tight = windows[-1]
    log(f"  REPORTED (kappa* fixed at the augmented apex {kfold:.6f}): "
        f"p = {tight['p_fixed']:.3f} at delta={tight['delta']:.3f}, "
        f"drifting to {windows[0]['p_fixed']:.3f} at delta={windows[0]['delta']:.3f}")
    log(f"  free-fit check: p -> {tight['p']:.3f}, free apex "
        f"{tight['kstar']:.5f} vs solved kappa* = {kfold:.5f}")
    log(f"  (the wide-window value p would give kappa*={kstar_wide:.5f}: biased, discarded)")
    np.savetxt(DATADIR / "two_dof_k3_nondeg_sqrt_windows.csv",
               np.array([[w["delta"], w["npts"], w["kstar"], w["amp"], w["p"], w["rms"],
                          w["amp_fixed"], w["p_fixed"], w["rms_fixed"]]
                         for w in windows]),
               delimiter=",",
               header=("delta,npts,kappa_star_fit,amplitude,exponent,rms,"
                       "amplitude_fixed,exponent_fixed,rms_fixed"), comments="")

    # ---------- (5) persist ----------
    np.savetxt(DATADIR / "two_dof_k3_nondeg_convergence.csv", np.array(conv_rows),
               delimiter=",", header="N,omega_star,beta1_star,abs_X3,abs_X1,cond_J", comments="")
    np.savetxt(DATADIR / "two_dof_k3_nondeg_saddlenode.csv", sn, delimiter=",",
               header="kappa,OmA,beta1A,X3A,X1A,OmB,beta1B,X3B,X1B", comments="")
    is_fold = np.zeros(len(om), dtype=int)
    is_fold[fold_idx] = 1
    np.savetxt(DATADIR / "two_dof_k3_nondeg_locus.csv",
               np.column_stack([om, be, ka, x3, x1, is_fold]),
               delimiter=",", header="omega,beta1,kappa,abs_X3,abs_X1,is_fold", comments="")
    status = np.array([[Om_star, b_star, KA0, F_FIX, KNL_FIX,
                        conv_seed["X3"], conv_seed["X1"], conv_seed["cond"],
                        A3_dop, float(rep.periodicity_residual),
                        tight["p"], tight["kstar"], kstar_wide,
                        tight["p_fixed"], windows[0]["p_fixed"],
                        max_x3_sn, float(ka[imax]), float(om[imax]), float(be[imax]),
                        len(om), float(x3.max())]])
    np.savetxt(DATADIR / "two_dof_k3_nondeg_status.csv", status, delimiter=",",
               header=("omega_star,beta1_star,kappa_fixed,F_fixed,kappa_nl_fixed,"
                       "abs_X3_seed,abs_X1_seed,cond_J_seed,A3_dop853,dop_periodicity_resid,"
                       "sqrt_exponent_tightwindow,kappa_star_tightwindow,"
                       "kappa_star_widewindow_biased,"
                       # the two exponents the manuscript quotes: kappa* pinned
                       # at the augmented apex, fitted in log-log
                       "sqrt_exponent_fixed_tightwindow,sqrt_exponent_fixed_widewindow,"
                       "max_abs_X3_saddlenode,kappa_fold_locus,"
                       "omega_fold_locus,beta1_fold_locus,locus_npts,max_abs_X3_locus"),
               comments="")

    # ---------- figure (authored at single-column width; no LaTeX rescaling) ----
    # No annotation box: the caption already states kappa*, the shrinking-window
    # exponent, the |X3| floor and cond(J). Repeating them on the axes is clutter.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=fs.column(0.62),
                                   constrained_layout=True)
    sA, sB = fs.series(0), fs.series(1)

    # (a) two arms of the exact zero merging at kappa* in (kappa, Omega).
    # Arm A runs flat along the top, arm B rises to meet it: the free corner is
    # lower-right, so the legend goes there and never sits on the data.
    axL.plot(sn[:, 0], sn[:, 1], **sA, label="arm A")
    axL.plot(sn[:, 0], sn[:, 5], **sB, label="arm B")
    axL.plot([kfold], [om[imax]], **fs.star(), label=r"fold $\kappa^\ast$")
    axL.axvline(kfold, color=fs.REFERENCE, ls=":", lw=0.7)
    axL.set_xlabel(r"$\kappa$")
    axL.set_ylabel(r"$\Omega$")
    # The two arms are far apart except near the apex, so they are labelled where
    # they run. A legend box in the lower-right corner sat on arm B.
    axL.annotate("arm A", xy=(sn[1, 0], sn[1, 1]), xytext=(1, 5),
                 textcoords="offset points", color=sA["color"], fontsize=7,
                 ha="left", va="bottom")
    axL.annotate("arm B", xy=(sn[2, 0], sn[2, 5]), xytext=(3, 2),
                 textcoords="offset points", color=sB["color"], fontsize=7,
                 ha="left", va="bottom")
    axL.annotate(r"fold $\kappa^\ast$", xy=(kfold, om[imax]), xytext=(-10, -20),
                 textcoords="offset points", fontsize=7, ha="right", va="top",
                 arrowprops=dict(arrowstyle="->", lw=0.5, color="0.3", shrinkB=4))
    fs.grid(axL)

    # (b) sqrt scaling of the branch separation about the *directly solved* apex.
    # log-log, so the exponent is a slope the reader can check by eye and the
    # asymptotic approach to 1/2 is visible rather than asserted. The reference
    # line is anchored on the tightest window, where the asymptotic law holds.
    m_tight = mask & (kfold - sn[:, 0] <= tight["delta"])
    C = float(np.mean(dsep[m_tight]
                      / np.sqrt(np.maximum(kfold - sn[m_tight, 0], 1e-12))))
    axR.loglog(kfold - sn[mask, 0], dsep[mask], ls="none", marker="o",
               color=fs.BLUE, ms=3.2, label="data")
    axR.loglog(kfold - sn[m_tight, 0], dsep[m_tight], ls="none", marker="o",
               mfc="none", mec=fs.NAVY, ms=6.5, mew=0.9,
               label=rf"$\delta\leq{tight['delta']:g}$")
    _kk = np.logspace(np.log10(0.6 * (kfold - sn[mask, 0]).min()),
                      np.log10(1.3 * (kfold - sn[mask, 0]).max()), 200)
    axR.loglog(_kk, C * np.sqrt(_kk), "-", color=fs.NAVY, lw=1.0,
               label=r"slope $1/2$")
    axR.set_xlabel(r"$\kappa^\ast-\kappa$")
    axR.set_ylabel(r"$|\Omega_A-\Omega_B|$")
    # data climbs lower-left -> upper-right corner to corner, so both off-diagonal
    # corners are free but narrow. A compact legend fits the upper-left one, which
    # lets the axis close just above the top point instead of carrying half a
    # decade of empty space to clear a full-size box.
    axR.set_ylim(top=dsep[mask].max() * 1.3)
    axR.legend(loc="upper left", fontsize=7, frameon=False, handlelength=1.2,
               handletextpad=0.4,
               labelspacing=0.25, borderpad=0.3, borderaxespad=0.3, framealpha=0.9)
    fs.grid(axR, which="both")

    fs.panel(axL, "(a)")
    fs.panel(axR, "(b)")
    fs.save(fig, FIGDIR, "k3_nondeg_exact_zero_fold")

    log("\nWrote:")
    for f in ("two_dof_k3_nondeg_convergence.csv", "two_dof_k3_nondeg_saddlenode.csv",
              "two_dof_k3_nondeg_locus.csv", "two_dof_k3_nondeg_status.csv"):
        log(f"  data/{f}")
    log("  figures/k3_nondeg_exact_zero_fold.{pdf,png}")


if __name__ == "__main__":
    main()
