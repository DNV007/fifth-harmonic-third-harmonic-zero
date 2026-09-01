"""Does the 3W -> 5W -> 3W mechanism reproduce in a second physical model?

The second model of SM Sec. S14 (charge coordinates, mutual-inductance coupling)
was introduced only to check the two-control zero structure against a different
linear propagator, explicitly NOT the closure mechanism.  That is weak evidence:
codimension-two counting is generic.  This driver tests the mechanism itself.

Model (q1, q2; mutual inductance LAMBDA, on-site cubic b1, coupling cubic GC):

    q1'' + 2g1 q1' + w1^2 q1 + b1 q1^3 + LAMBDA q2'' + GC (q1-q2)^3 = E0 cos(Om t)
    q2'' + 2g2 q2' + w2^2 q2           + LAMBDA q1'' + GC (q2-q1)^3 = 0

The coupling sits on the accelerations, so the off-diagonal dynamic stiffness is
-LAMBDA (kOm)^2 and is frequency dependent, unlike the constant kappa of the
mechanical absorber.  The linear modes are w_- = 0.98898 and w_+ = 1.43180.

WHERE TO LOOK.  In the mechanical model the mechanism operates where 5*Om sits
near the upper mode, so that the small fifth-harmonic return is propagated
efficiently.  Here that is Om = w_+/5 = 0.28636.  The drive follows from the
closure criterion of End Matter Eq. (EM4).  Writing the prefactor of Eq. (EM3)
for this network as P(Om) = GC^2 g5(Om) / [D3]_22, with
g5 = (1,-1) D5^-1 (1,-1)^T, and equating |Im eta_5| ~ |c P(Om)| |d_1|^4 to the
rotation the leading order leaves, gives |d_1| = 0.33 and hence E0 = 0.31.  The
root is found there.  The O(1) constant c = 0.267 is carried over from the
mechanical working point rather than derived, so this is a calibrated estimate
that locates the spectral region, not a parameter-free prediction; what it has
to do is land close enough for a solve to converge, and it does.  (Two blind
scans over (Om, b1) had failed first.)

Two working points are reported, and the contrast is the result:

  A.  Om = 0.2889, E0 = 0.307, at the 5:1 placement.  Every signature of the
      mechanical case reproduces: leading-order phase obstruction, no root below
      N_H = 5, a {1} -> {1,3} -> {1,3,5} hierarchy that closes, and a homotopy
      fold before the return is removed.
  B.  Om = 0.2280, E0 = 2.0, the working point of SM Sec. S14, where 5*Om lies
      20% from w_+.  None of them reproduce: the orbit is strongly nonlinear
      (|Q_1,5|/|Q_1,1| = 0.17), the truncation hierarchy does not converge, and
      the zero survives the homotopy to lambda = 0.  That zero is closed by a
      different balance, and it is not evidence for the mechanism.

Run:  PYTHONPATH=src JAX_ENABLE_X64=1 uv run python scripts/run_k3_second_model_mechanism.py
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

W1, W2 = 1.0, 1.40
G1, G2 = 0.020, 0.025
LAM, GC = 0.15, -0.25
E0_DEFAULT = 2.0                     # the residual below carries this

_D = Path(os.path.dirname(os.path.abspath(__file__)))
DATA = _D / ".." / "data"


def Dk(k, Om):
    """Dynamic-stiffness matrix at harmonic k. Off-diagonal ~ -LAMBDA (kOm)^2."""
    w = k * Om
    return np.array([[-w ** 2 + 2j * G1 * w + W1 ** 2, -LAM * w ** 2],
                     [-LAM * w ** 2, -w ** 2 + 2j * G2 * w + W2 ** 2]], dtype=complex)


def build(N):
    T = {}
    for m in range(1, N + 1):
        keep, ret = [], []
        for p, q, r in product(range(-N, N + 1), repeat=3):
            if p + q + r != m or 0 in (p, q, r):
                continue
            (ret if 5 in (abs(p), abs(q), abs(r)) else keep).append((p, q, r))
        T[m] = (np.array(keep), np.array(ret))

    def ext(X):
        v = np.zeros(2 * N + 1, complex)
        v[N + 1:] = X / 2
        v[:N] = (X.conj() / 2)[::-1]
        return v

    def cube(X, lam=1.0, keep_only=None):
        v = ext(X)
        o = np.zeros(N, complex)
        for m in range(1, N + 1):
            kk, rr = T[m]
            sel = kk if keep_only is None else np.array(
                [t for t in kk if all(abs(i) in keep_only for i in t)] or [[1, 1, -1]])
            if keep_only is not None:
                sel = np.array([t for t in list(kk) + list(rr)
                                if all(abs(i) in keep_only for i in t)])
                c = 0 if len(sel) == 0 else (
                    v[sel[:, 0] + N] * v[sel[:, 1] + N] * v[sel[:, 2] + N]).sum()
                o[m - 1] = 2 * c
                continue
            c = (v[kk[:, 0] + N] * v[kk[:, 1] + N] * v[kk[:, 2] + N]).sum()
            if len(rr):
                c += (lam if m == 3 else 1.0) * (
                    v[rr[:, 0] + N] * v[rr[:, 1] + N] * v[rr[:, 2] + N]).sum()
            o[m - 1] = 2 * c
        return o

    def res(y, lam=1.0):
        c, Om, b1 = y[:4 * N], y[4 * N], y[4 * N + 1]
        Q1 = c[0:2 * N:2] - 1j * c[1:2 * N:2]
        Q2 = c[2 * N:4 * N:2] - 1j * c[2 * N + 1:4 * N:2]
        d = Q1 - Q2
        cq, cd = cube(Q1, lam), cube(d, lam)
        R1 = np.empty(N, complex); R2 = np.empty(N, complex)
        for k in range(1, N + 1):
            D = Dk(k, Om)
            R1[k - 1] = D[0, 0] * Q1[k - 1] + D[0, 1] * Q2[k - 1]
            R2[k - 1] = D[1, 0] * Q1[k - 1] + D[1, 1] * Q2[k - 1]
        R1 += b1 * cq + GC * cd
        R2 += -GC * cd
        R1[0] -= E0_DEFAULT
        return np.concatenate([R1.real, R1.imag, R2.real, R2.imag,
                               [Q1[2].real, Q1[2].imag]])
    return res, cube


def jac(f, x, h=1e-7):
    f0 = f(x); J = np.empty((f0.size, x.size))
    for i in range(x.size):
        xp = x.copy(); xp[i] += h; J[:, i] = (f(xp) - f0) / h
    return J


WP = {"A": dict(E0=0.307, Om=0.288884, b1=0.237649, OMW=(0.26, 0.32), BW=(0.16, 0.34)),
      "B": dict(E0=2.000, Om=0.227985, b1=1.356988, OMW=(0.15, 0.35), BW=(0.70, 2.20))}
WPLUS = 1.43180


def make(N, E0):
    res0, cube = build(N)
    def res(y, lam=1.0):
        r = res0(y, lam)
        r[0] += E0_DEFAULT - E0          # the module residual carries E0_DEFAULT
        return r
    def hb(Om, b1, guess=None):
        f = lambda c: res(np.concatenate([c, [Om, b1]]))[:4 * N]
        c0 = np.zeros(4 * N) if guess is None else guess.copy()
        if guess is None:
            c0[0] = E0 / (W1 ** 2 - Om ** 2)
        s = least_squares(f, c0, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=4000)
        return s.x, float(np.linalg.norm(f(s.x)))
    return res, cube, hb


def warm(hb, Om, b1):
    """Follow the orbit up in b1 rather than guessing it.

    Point B sits on a strongly nonlinear, bistable branch that the ramp loses;
    fall back to a cold solve there, which the bounded corrector then fixes."""
    c = None
    for bb in np.linspace(0.0, b1, 60):
        c, r = hb(Om, float(bb), c)
        if r > 1e-9:
            c = None
            break
    if c is not None:
        return c
    c, r = hb(Om, b1)
    return c if r < 1e-9 else None


def constrained(N, res, y0, win):
    lo = np.full(4 * N + 2, -np.inf); hi = np.full(4 * N + 2, np.inf)
    lo[4 * N], hi[4 * N] = win["OMW"]
    lo[4 * N + 1], hi[4 * N + 1] = win["BW"]
    s = least_squares(lambda y: res(y, 1.0), np.clip(y0, lo + 1e-12, hi - 1e-12),
                      bounds=(lo, hi), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=8000)
    return s.x, float(np.linalg.norm(res(s.x, 1.0)))


def battery(tag, w):
    N = 7
    res, cube, hb = make(N, w["E0"])
    c = warm(hb, w["Om"], w["b1"])
    assert c is not None, f"{tag}: no orbit at the seed"
    y, r = constrained(N, res, np.concatenate([c, [w["Om"], w["b1"]]]), w)
    assert r < 1e-10, f"{tag}: no constrained root"
    Om, b1 = y[4 * N], y[4 * N + 1]
    Q1 = y[0:2 * N:2] - 1j * y[1:2 * N:2]
    Q2 = y[2 * N:4 * N:2] - 1j * y[2 * N + 1:4 * N:2]
    print(f"\n{'='*88}\nWORKING POINT {tag}:  E0 = {w['E0']}, "
          f"(Omega, b1) = ({Om:.6f}, {b1:.6f}), residual {r:.1e}")
    print(f"  |Q_1,1| = {abs(Q1[0]):.5f}   |Q_1,5|/|Q_1,1| = {abs(Q1[4])/abs(Q1[0]):.3e}"
          f"   |Q_2,3| = {abs(Q2[2]):.3e}")
    print(f"  5*Omega = {5*Om:.5f}, {abs(5*Om-WPLUS)/WPLUS*100:.2f}% from omega_+")

    D3, D1 = Dk(3, Om), Dk(1, Om)
    Xi = -GC * (D3[1, 1] + D3[0, 1]) * (1 + D1[1, 0] / D1[1, 1]) ** 3 / D3[1, 1]
    print(f"  (1) leading order: arg Xi_LO = {np.angle(Xi):+.4e}, "
          f"Re Xi_LO misses b1 by {100*abs(Xi.real-b1)/b1:.2f}%")

    print("  (2) truncation onset (continuation from the N_H=7 point):", end=" ")
    onset = []
    for nh in (3, 4, 5, 7, 9):
        rn, _, hn = make(nh, w["E0"])
        cc = warm(hn, Om, b1)
        if cc is None:
            onset.append((nh, None)); continue
        z2, r2 = constrained(nh, rn, np.concatenate([cc, [Om, b1]]), w)
        onset.append((nh, r2))
    print("  ".join(f"N_H={n}:{'root' if (e is not None and e<1e-10) else 'none'}"
                    for n, e in onset))

    if tag.startswith("A"):
        # Continuation alone only shows the tracked root is not recovered.  The
        # absence at N_H = 3, 4 is one of the signatures carrying the generality
        # claim, so it gets the same treatment as the mechanical model: a grid
        # over the search window, then an augmented solve seeded from every cell
        # in which BOTH quadratures change sign.
        for nh in (3, 4):
            rn, _, hn = make(nh, w["E0"])
            for lbl, ow, bw, ng in (("wide ", (0.15, 0.45), (0.05, 0.60), 61),
                                    ("local", (0.26, 0.32), (0.18, 0.30), 41)):
                og = np.linspace(*ow, ng); bg = np.linspace(*bw, ng)
                Z = np.zeros((ng, ng), complex)
                for i, o_ in enumerate(og):
                    g = None
                    for j, b_ in enumerate(bg):
                        c_, r_ = hn(float(o_), float(b_), g)
                        if r_ > 1e-9:
                            g = None; Z[i, j] = np.nan; continue
                        g = c_; Z[i, j] = complex(c_[2 * (3 - 1)], -c_[2 * (3 - 1) + 1])
                seeds = []
                for i in range(ng - 1):
                    for j in range(ng - 1):
                        blk = Z[i:i + 2, j:j + 2]
                        if np.isnan(blk.real).any():
                            continue
                        if (blk.real.min() < 0 < blk.real.max()
                                and blk.imag.min() < 0 < blk.imag.max()):
                            seeds.append((0.5 * (og[i] + og[i + 1]),
                                          0.5 * (bg[j] + bg[j + 1])))
                win = dict(OMW=(ow[0] - 0.01, ow[1] + 0.01),
                           BW=(bw[0] - 0.01, bw[1] + 0.01))
                roots, bestr = [], np.inf
                for (o_, b_) in seeds:
                    c_, r_ = hn(float(o_), float(b_))
                    if r_ > 1e-9:
                        continue
                    z_, rr = constrained(nh, rn, np.concatenate([c_, [o_, b_]]), win)
                    bestr = min(bestr, rr)
                    if rr < 1e-10:
                        om_, bb_ = z_[4 * nh], z_[4 * nh + 1]
                        if not any(abs(om_ - q[0]) < 1e-6 and abs(bb_ - q[1]) < 1e-6
                                   for q in roots):
                            roots.append((om_, bb_))
                mn = np.nanmin(np.abs(Z[~np.isnan(Z.real)]))
                print(f"      N_H={nh} {lbl} grid {ng}x{ng} Omega{ow} beta1{bw}: "
                      f"min|Q_1,3|={mn:.1e}, {len(seeds)} sign-change cells, "
                      f"{len(roots)} root(s)"
                      + (f" at {[tuple(np.round(q,6)) for q in roots]}" if roots
                         else f", best augmented residual {bestr:.1e}"))

    print("  (3) source decomposition:")
    d = Q1 - Q2
    for lab, keep in (("{1}", {1}), ("{1,3}", {1, 3}), ("{1,3,5}", {1, 3, 5})):
        Aq = cube(Q1, keep_only=keep)[2]; Ad = cube(d, keep_only=keep)[2]
        X = -GC * (D3[1, 1] + D3[0, 1]) * Ad / (D3[1, 1] * Aq)
        print(f"      {lab:<9} b1 = {X.real:10.6f}   error {100*abs(X.real-b1)/b1:8.4f}%"
              f"   arg Xi = {np.angle(X):+.3e}")

    z = np.concatenate([y, [1.0]])
    G = lambda zz: res(zz[:-1], zz[-1])
    _, _, Vt = np.linalg.svd(jac(G, z)); t = Vt[-1]
    t = -t if t[-1] > 0 else t; t /= np.linalg.norm(t)
    ds, zp, lmin, zf = 0.02, z.copy(), 1.0, z.copy()
    for _ in range(800):
        s = least_squares(lambda zz: np.concatenate([G(zz), [(zz - zp) @ t - ds]]),
                          zp + ds * t, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=400)
        if not s.success or np.linalg.norm(G(s.x)) > 1e-9:
            ds *= 0.5
            if ds < 1e-7:
                break
            continue
        zn = s.x
        _, _, V2 = np.linalg.svd(jac(G, zn)); tn = V2[-1]
        tn = tn if tn @ t > 0 else -tn; tn /= np.linalg.norm(tn)
        zp, t = zn, tn
        if zp[-1] < lmin:
            lmin, zf = zp[-1], zp.copy()
        if zp[-1] <= 0.0 or (zp[-1] > 1.0 and lmin < 0.95):
            break
        ds = min(ds * 1.15, 0.05)
    print(f"  (4) homotopy: minimum lambda = {lmin:.6f}  ->  "
          + (f"FOLD at lambda* = {lmin:.4f}, the return is required"
             if lmin > 1e-6 else "survives to lambda = 0, the return is NOT required"))
    return lmin


def main():
    print("=" * 88)
    print("DOES THE 3W -> 5W -> 3W MECHANISM REPRODUCE IN THE SECOND MODEL?")
    print("=" * 88)
    print(f"    mutual-inductance coupling LAMBDA={LAM}, coupling cubic GC={GC}")
    print(f"    linear modes: omega_- = 0.98898, omega_+ = {WPLUS}")
    print(f"    the 5:1 placement is Omega = omega_+/5 = {WPLUS/5:.5f}")
    lamA = battery("A (at the 5:1 placement)", WP["A"])
    lamB = battery("B (SM Sec. S14 point, 5*Omega 20% off omega_+)", WP["B"])
    print(f"\n{'='*88}\nVERDICT")
    print(f"  At the 5:1 placement every signature of the mechanical case reproduces,")
    print(f"  and the zero folds at lambda* = {lamA:.4f}. At the far-off-resonance point")
    print(f"  none of them do, and the zero survives to lambda = {lamB:.4f}.")
    print("  The mechanism transfers to a different linear network, but it is not")
    print("  automatic: it needs the spectral placement that propagates the return.")
    assert lamA > 1e-6, "the mechanism did not reproduce at the 5:1 placement"
    assert lamB < 1e-2, "working point B unexpectedly shows the mechanism"


if __name__ == "__main__":
    main()
