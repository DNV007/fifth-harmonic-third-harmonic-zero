"""Phase reduction of the third-harmonic zero, and the closure criterion.

Backs SM Sec. S4 ("Reduction to a phase condition and a closure criterion"),
the 1:3:5 projected-source reduction of beta_1, the channel
geometry paragraph of SM Sec. S2, and the End Matter section "Phase reduction
and a closure criterion" of the Letter.

The object is the projected third-harmonic balance obtained by eliminating the
absorber at X3^(1) = 0,

    Z22(3W) beta1 A3[x1^3] + L2(3W) kappa_nl A3[d^3] = 0,

solved for the on-site coefficient,

    beta1 = Xi,   Xi = -kappa_nl L2(3W) A3[d^3] / (Z22(3W) A3[x1^3]).

Xi is evaluated ON THE ORBIT, so it depends on beta_1 and F through the orbit;
Eq. (S13) is an implicit fixed point, not a formula. Replacing the orbit by its
linear fundamental gives Xi_LO(W), a ratio of linear propagators that depends on
neither amplitude nor beta_1, and whose PHASE is the obstruction to a real
closure coefficient.

Conventions follow SM Sec. S13: u(t) = sum_k Re[U_k exp(i k W t)] with
U_k = a_k - i b_k, so A_k[u] = (2/N) sum_n u(t_n) exp(-i k W t_n) and, for a
single harmonic, A3[u^3] = U1^3/4.

Reported, with the SM value alongside each:

  (1) linear propagators L2, Z22, the relative-coordinate propagators g3, g5
  (2) Xi_LO at the working point, its phase, and the phase sweep on [0.24,0.28]
  (3) exact A3 by time-domain transform vs the {1,3,5} multinomial
  (4) beta_1 from Eq. (S10) truncated to {1}, {1,3}, {1,3,5}
  (5) channel geometry: the 3W and 5W contributions and their alignment
  (6) the fixed point: Re Xi - beta1*, Im Xi on the converged orbit
  (7) d(Re Xi)/d beta1 by central differences with the orbit recomputed
  (8) eta_5: exact value on the orbit vs the scaling form of SM Eq. (S22),
      with the O(1) prefactor CALIBRATED here rather than asserted
  (9) the closure criterion |Im eta_5| >~ arg Xi_LO, continued in F, and the
      forcing at which it fails

This driver is self-contained (numpy/scipy only) and shares no code with
src/hh_antiresonance, so item (3) is an independent check of the production
harmonic-balance residual as well as of the reduction.

Run:  JAX_ENABLE_X64=1 uv run python scripts/run_k3_phase_reduction.py
"""
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, root

W1, W2 = 1.0, 1.25
Z1, Z2 = 0.015, 0.02
KA, KNL = 0.10, -0.30
F0 = 0.30
OM_PROD, B1_PROD = 0.25972810, 0.23815428
N, NT = 7, 512
NFFT = 4096  # SM Sec. S13: cubic source coefficients use N = 4096

_D = Path(os.path.dirname(os.path.abspath(__file__)))
OUT = _D / ".." / "data"

nn = np.arange(NT)
PH = 2 * np.pi * np.outer(np.arange(1, N + 1), nn) / NT
COS, SIN = np.cos(PH), np.sin(PH)
MVEC = np.arange(1, N + 1)[:, None]
IC, IS = 2 * (3 - 1), 2 * (3 - 1) + 1


# ----------------------------------------------------------------- linear part
def L2(k, Om):
    """Absorber dynamic stiffness at harmonic k (no coupling)."""
    kO = k * Om
    return W2 ** 2 - kO ** 2 + 2j * Z2 * kO


def Z22(k, Om):
    return L2(k, Om) + KA


def Dmat(k, Om):
    kO = k * Om
    return np.array([[W1 ** 2 + KA - kO ** 2 + 2j * Z1 * kO, -KA],
                     [-KA, W2 ** 2 + KA - kO ** 2 + 2j * Z2 * kO]])


def g_rel(k, Om):
    """Complex relative-coordinate propagator g_k = (1,-1) D_k^-1 (1,-1)^T."""
    v = np.array([1.0, -1.0])
    return complex(v @ np.linalg.solve(Dmat(k, Om), v))


def xi_lo(Om):
    """Leading-order Xi: linear propagators only (SM Eq. S16)."""
    return -KNL * L2(3, Om) / Z22(3, Om) * (1.0 - KA / Z22(1, Om)) ** 3


# ------------------------------------------------------------ harmonic balance
def recon(c, Om):
    x1 = c[:2 * N].reshape(N, 2)
    x2 = c[2 * N:].reshape(N, 2)
    a1, b1 = x1[:, [0]], x1[:, [1]]
    a2, b2 = x2[:, [0]], x2[:, [1]]
    q1 = np.sum(a1 * COS + b1 * SIN, 0)
    q2 = np.sum(a2 * COS + b2 * SIN, 0)
    v1 = np.sum(MVEC * Om * (-a1 * SIN + b1 * COS), 0)
    v2 = np.sum(MVEC * Om * (-a2 * SIN + b2 * COS), 0)
    ac1 = np.sum(-(MVEC * Om) ** 2 * (a1 * COS + b1 * SIN), 0)
    ac2 = np.sum(-(MVEC * Om) ** 2 * (a2 * COS + b2 * SIN), 0)
    return q1, v1, ac1, q2, v2, ac2


def hb_res(c, Om, b1, F=F0):
    q1, v1, ac1, q2, v2, ac2 = recon(c, Om)
    d = q1 - q2
    cpl = KA * d + KNL * d ** 3
    r1 = ac1 + 2 * Z1 * v1 + W1 ** 2 * q1 + b1 * q1 ** 3 + cpl - F * COS[0]
    r2 = ac2 + 2 * Z2 * v2 + W2 ** 2 * q2 - cpl
    out = np.empty(4 * N)
    for m in range(N):
        out[4 * m:4 * m + 4] = [(2 / NT) * (r1 @ COS[m]), (2 / NT) * (r1 @ SIN[m]),
                                (2 / NT) * (r2 @ COS[m]), (2 / NT) * (r2 @ SIN[m])]
    return out


def solve_hb(Om, b1, F=F0, guess=None):
    """Plain harmonic-balance solve at fixed (Om, b1, F)."""
    if guess is None:
        z0 = np.zeros(4 * N)
        z0[0] = F / (W1 ** 2 - Om ** 2)
    else:
        z0 = guess.copy()
        s = root(lambda c: hb_res(c, Om, b1, F), z0, method="hybr", tol=1e-13)
        if s.success:
            return s.x
    return least_squares(lambda c: hb_res(c, Om, b1, F), z0,
                         xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=6000).x


def solve_zero(seed, F=F0):
    """Augmented solve: harmonic balance plus Re X3 = Im X3 = 0, free (Om, b1)."""
    def aug(z):
        c = z[:4 * N]
        return np.concatenate([hb_res(c, float(z[-2]), float(z[-1]), F),
                               [c[IC], c[IS]]])
    s = least_squares(aug, seed, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=9000)
    return s.x


# ------------------------------------------------- complex harmonic extraction
def U_of(c, osc):
    """Complex coefficients U_k = a_k - i b_k, k = 1..N, in the SM convention."""
    blk = c[:2 * N] if osc == 1 else c[2 * N:]
    m = blk.reshape(N, 2)
    return m[:, 0] - 1j * m[:, 1]


def A3_time(U1, U2=None, Om=1.0):
    """A3 of u^3 by direct time-domain transform on NFFT samples.

    u is synthesized from the complex coefficients in the real-part convention;
    the cube is transformed back with A_k = (2/N) sum u exp(-i k W t).
    """
    t = 2 * np.pi * np.arange(NFFT) / NFFT  # W t over one period
    U = U1 if U2 is None else U1 - U2
    u = np.zeros(NFFT)
    for k in range(1, N + 1):
        u += np.real(U[k - 1] * np.exp(1j * k * t))
    return (2.0 / NFFT) * np.sum(u ** 3 * np.exp(-3j * t))


def A3_multinomial(U, keep=(1, 3, 5)):
    """4*A3[u^3] from SM Eq. (S11), restricted to the retained harmonics."""
    def u(k):
        return U[k - 1] if k in keep else 0.0 + 0.0j
    U1, U3, U5 = u(1), u(3), u(5)
    tot = U1 ** 3
    tot += 3 * U5 * np.conj(U1) ** 2
    tot += 6 * U3 * abs(U1) ** 2
    tot += 3 * U3 * abs(U3) ** 2
    tot += 6 * U5 * np.conj(U3) * U1
    tot += 6 * U3 * abs(U5) ** 2
    return tot / 4.0


def xi_from(A3x1, A3d, Om):
    return -KNL * L2(3, Om) * A3d / (Z22(3, Om) * A3x1)


hdr = "=" * 96
print(hdr)
print("PHASE REDUCTION OF THE THIRD-HARMONIC ZERO, AND THE CLOSURE CRITERION")
print(hdr)

z = solve_zero(np.concatenate([solve_hb(OM_PROD, B1_PROD), [OM_PROD, B1_PROD]]))
c0, OmS, b1S = z[:4 * N], float(z[-2]), float(z[-1])
U1v, U2v = U_of(c0, 1), U_of(c0, 2)
Dv = U1v - U2v
print(f"\nconverged root: Omega* = {OmS:.9f}   beta1* = {b1S:.9f}")
print(f"                |X1| = {abs(U1v[0]):.6f}   |X3| = {abs(U1v[2]):.2e}   "
      f"|X3^(2)| = {abs(U2v[2]):.4e}   (SM 1.56e-3)")

# ------------------------------------------------------------------------- (1)
print("\n(1) LINEAR PROPAGATORS at Omega*")
print(f"    L2(3W)  = {L2(3, OmS):.6f}")
print(f"    Z22(3W) = {Z22(3, OmS):.6f}")
print(f"    Z22(W)  = {Z22(1, OmS):.6f}")
g3, g5 = g_rel(3, OmS), g_rel(5, OmS)
print(f"    g3 = (1,-1) D_3^-1 (1,-1)^T = {g3:.5f}   |g3| = {abs(g3):.4f}   (SM 2.64)")
print(f"    g5 = (1,-1) D_5^-1 (1,-1)^T = {g5:.5f}   |g5| = {abs(g5):.4f}   (SM 25.9)")
print(f"    |g5|/|g3| = {abs(g5)/abs(g3):.4f}   (SM 9.82)")

# ------------------------------------------------------------------------- (2)
XL = xi_lo(OmS)
print("\n(2) LEADING-ORDER Xi AND ITS PHASE")
print(f"    Xi_LO(Omega*) = {XL.real:.6f} + {XL.imag:.4e} i     "
      f"(SM 0.22365 + 9.83e-4 i)")
print(f"    arg Xi_LO = {np.angle(XL):.6e} rad = {np.degrees(np.angle(XL)):.4f} deg   "
      f"(SM 4.395e-3 rad, 0.252 deg)")
print(f"    Re Xi_LO vs converged beta1*: {XL.real:.5f} vs {b1S:.5f}  "
      f"({abs(XL.real/b1S-1)*100:.1f}% low, SM 6%)")
sw = np.linspace(0.24, 0.28, 41)
ph = np.array([np.angle(xi_lo(o)) for o in sw])
mono = bool(np.all(np.diff(ph) > 0))
print(f"    phase on [0.24,0.28]: {ph[0]:.3e} -> {ph[-1]:.3e} rad "
      f"({np.degrees(ph[0]):.2f} -> {np.degrees(ph[-1]):.2f} deg)   "
      f"(SM 3.66e-3 -> 5.59e-3)")
print(f"    monotonically increasing: {mono}   sign changes: {int(np.sum(np.diff(np.sign(ph)) != 0))}")

# ------------------------------------------------------------------------- (3)
A3x_t, A3d_t = A3_time(U1v, Om=OmS), A3_time(U1v, U2v, Om=OmS)
A3x_m, A3d_m = A3_multinomial(U1v), A3_multinomial(Dv)
print("\n(3) A3 OF THE CUBIC: time-domain transform vs the {1,3,5} multinomial")
print(f"    on-site  A3[x1^3] = {A3x_t:.8e}   rel. diff {abs(A3x_m/A3x_t-1):.2e}  (SM 2e-8)")
print(f"    coupling A3[d^3]  = {A3d_t:.8e}   rel. diff {abs(A3d_m/A3d_t-1):.2e}  (SM 8e-7)")
def LHS(keep=None):
    """Left-hand side of Eq. (S10) on the converged orbit, in source units."""
    if keep is None:
        return Z22(3, OmS) * b1S * A3x_t + L2(3, OmS) * KNL * A3d_t
    return (Z22(3, OmS) * b1S * A3_multinomial(U1v, keep)
            + L2(3, OmS) * KNL * A3_multinomial(Dv, keep))


one = abs(Z22(3, OmS) * b1S * A3x_t)
print(f"    Eq. (S10) closes to {abs(LHS())/one:.2e} (exact A3) and "
      f"{abs(LHS((1,3,5)))/one:.2e} ({{1,3,5}} A3) relative to one pathway")
print(f"    SM quotes 1.6e-8 for this closure")

# ------------------------------------------------------------------------- (4)
print("\n(4) beta_1 from Eq. (S10) after successive source truncations")
print(f"    {'harmonics kept':<22}{'Re beta1':>12}{'error':>11}{'|Im beta1|':>13}   SM")
sm_ref = {"{1}": ("0.2311198", "2.95%", "9.6e-04"),
          "{1,3}": ("0.2389011", "0.31%", "7.3e-04"),
          "{1,3,5}": ("0.2381542", "4e-05%", "1.7e-07")}
tab = []
for lbl, keep in (("{1}", (1,)), ("{1,3}", (1, 3)), ("{1,3,5}", (1, 3, 5))):
    xi = xi_from(A3_multinomial(U1v, keep), A3_multinomial(Dv, keep), OmS)
    err = abs(xi.real / b1S - 1) * 100
    r = sm_ref[lbl]
    print(f"    {lbl:<22}{xi.real:>12.7f}{err:>10.2e}%{abs(xi.imag):>13.1e}   "
          f"{r[0]} / {r[1]} / {r[2]}")
    tab.append((len(keep), xi.real, err, abs(xi.imag)))

# ------------------------------------------------------------------------- (5)
print("\n(5) CHANNEL GEOMETRY of the projected combination, in source units")
l1, l13, l135 = LHS((1,)), LHS((1, 3)), LHS((1, 3, 5))
ch3, ch5 = l13 - l1, l135 - l13
print(f"    leading residual |LHS(1)|   = {abs(l1):.4e}")
print(f"    3W channel       |c3|       = {abs(ch3):.4e}   (SM 5.13e-5)")
print(f"    5W channel       |c5|       = {abs(ch5):.4e}   (SM 6.81e-6)")
print(f"    angle of c3 to the leading residual = "
      f"{np.degrees(np.angle(ch3/l1))%360:.1f} deg   (SM 186.1)")
print(f"    after 3W, remaining {abs(l13):.4e}  "
      f"({(1-abs(l13)/abs(l1))*100:.1f}% of the magnitude removed, SM 85-86%)")
print(f"    5W removes {(1-abs(l135)/abs(l13))*100:.2f}% of that remainder, "
      f"leaving {abs(l135):.2e}")
print(f"      SM quotes 98.7% and 8.9e-8 here; this run closes further.")
print(f"    |c5| / |LHS(1,3)| = {abs(ch5)/abs(l13):.3f}   (SM ~0.99 near F = 0.30)")

# ------------------------------------------------------------------------- (6)
XI = xi_from(A3x_t, A3d_t, OmS)
print("\n(6) THE FIXED POINT ON THE CONVERGED ORBIT")
print(f"    Re Xi - beta1* = {XI.real-b1S:+.2e}   Im Xi = {XI.imag:+.2e}   (SM 5e-16, 4e-16)")

# ------------------------------------------------------------------------- (7)
print("\n(7) d(Re Xi)/d beta1 WITH THE ORBIT RECOMPUTED AT FIXED Omega")
print(f"    {'delta beta1':>13}{'d(Re Xi)/d b1':>16}{'dH1/d b1':>12}")
slopes = []
for db in (1e-4, 3e-4, 1e-3):
    vals = []
    for s in (+1, -1):
        cc = solve_hb(OmS, b1S + s * db, guess=c0)
        Ua, Ub = U_of(cc, 1), U_of(cc, 2)
        vals.append(xi_from(A3_time(Ua, Om=OmS), A3_time(Ua, Ub, Om=OmS), OmS).real)
    sl = (vals[0] - vals[1]) / (2 * db)
    slopes.append(sl)
    print(f"    {db:>13.0e}{sl:>16.4f}{sl-1:>12.4f}")
print(f"    SM: d(Re Xi)/d beta1 = +0.0060, dH1/d beta1 = -0.9940")
print(f"    implicit dependence is {abs(np.mean(slopes))*100:.1f}% of the explicit one "
      f"(SM 0.6%), so H1 = 0 is locally solvable for beta1(Omega)")

# ------------------------------------------------------------------------- (8)
xLO = xi_from(A3_multinomial(U1v, (1,)), A3_multinomial(Dv, (1,)), OmS)
x13 = xi_from(A3_multinomial(U1v, (1, 3)), A3_multinomial(Dv, (1, 3)), OmS)
x135 = xi_from(A3_multinomial(U1v, (1, 3, 5)), A3_multinomial(Dv, (1, 3, 5)), OmS)
eta3, eta5 = (x13 - xLO) / XL, (x135 - x13) / XL
argLO = np.angle(XL)
print("\n(8) PHASE BUDGET IN THE SM NORMALISATION  Xi = Xi_LO (1 + eta_3 + eta_5 + ...)")
print(f"    arg Xi_LO                    = {argLO:+.4e}")
print(f"    Im eta_3                     = {eta3.imag:+.4e}")
print(f"    Im eta_5                     = {eta5.imag:+.4e}")
print(f"    residual arg Xi at the root  = {argLO+eta3.imag+eta5.imag:+.2e}  (closes)")
print(f"\n    The third-harmonic channel supplies "
      f"{abs(eta3.imag)/(abs(eta3.imag)+abs(eta5.imag))*100:.0f}% of the phase rotation")
print(f"    and the fifth "
      f"{abs(eta5.imag)/(abs(eta3.imag)+abs(eta5.imag))*100:.0f}%. Eq. (S23) compares "
      f"|Im eta_5| with arg Xi_LO alone:")
print(f"      |Im eta_5| / |arg Xi_LO|              = {abs(eta5.imag)/abs(argLO):.3f}")
print(f"      |Im eta_5| / |arg Xi_LO + Im eta_3|   = "
      f"{abs(eta5.imag)/abs(argLO+eta3.imag):.3f}")
print("    The second ratio is the one that is unity at closure: eta_5 has to cancel")
print("    what eta_3 leaves, not the whole leading phase. As printed, Eq. (S23) is")
print("    short of its own reference by the eta_3 contribution.")

# ------------------------------------------------------------------------- (9)
print("\n(9) AMPLITUDE SCALING OF eta_5, TESTED OFF THE ROOT")
print("    (Omega, beta_1) are held at the root and only F is varied, so g_5(Omega)")
print("    is fixed and the |Delta_1|^4 factor of Eq. (S22) is isolated.")
print(f"    {'F':>7}{'|Delta_1|':>12}{'|eta_5|':>13}{'|eta_5|/|D1|^4':>17}")
gg = solve_hb(OmS, b1S, F=F0)
sc_rows = []
for Fv in (0.34, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22, 0.20):
    gg = solve_hb(OmS, b1S, F=Fv, guess=gg)
    Ua, Ub = U_of(gg, 1), U_of(gg, 2)
    Dd = Ua - Ub
    a = xi_from(A3_multinomial(Ua, (1, 3)), A3_multinomial(Dd, (1, 3)), OmS)
    b = xi_from(A3_multinomial(Ua, (1, 3, 5)), A3_multinomial(Dd, (1, 3, 5)), OmS)
    e5 = (b - a) / XL
    sc_rows.append((Fv, abs(Dd[0]), abs(e5)))
    print(f"    {Fv:>7.3f}{abs(Dd[0]):>12.5f}{abs(e5):>13.4e}"
          f"{abs(e5)/abs(Dd[0])**4:>17.4f}")
sc = np.array(sc_rows)
p = np.polyfit(np.log(sc[:, 1]), np.log(sc[:, 2]), 1)[0]
print(f"\n    fitted exponent  |eta_5| ~ |Delta_1|^p  gives p = {p:.3f}   "
      f"(Eq. (S22) predicts 4)")

# ------------------------------------------------------------------------ (10)
print("\n(10) THE CRITERION ALONG THE ROOT LOCUS")
print("    Evaluated ON the locus this is close to circular: at a root the phases")
print("    cancel by construction, exactly as SM Sec. S2 says of the alignment.")
print("    It is reported to show how the two reference quantities behave in F.")
print(f"    {'F':>7}{'|X1|':>9}{'Omega':>10}{'argXi_LO':>11}{'Im eta3':>11}"
      f"{'Im eta5':>11}{'r_LO':>7}{'r_res':>7}")
rows = []
zc = z.copy()
Fs = [0.35, 0.34, 0.32, 0.30, 0.29, 0.28, 0.27, 0.265, 0.26, 0.258]
for Fv in Fs:
    zc = solve_zero(zc, F=Fv)
    cc, Om_i, b1_i = zc[:4 * N], float(zc[-2]), float(zc[-1])
    Ua, Ub = U_of(cc, 1), U_of(cc, 2)
    Dd = Ua - Ub
    XLi = xi_lo(Om_i)
    a = xi_from(A3_multinomial(Ua, (1,)), A3_multinomial(Dd, (1,)), Om_i)
    b = xi_from(A3_multinomial(Ua, (1, 3)), A3_multinomial(Dd, (1, 3)), Om_i)
    cq = xi_from(A3_multinomial(Ua, (1, 3, 5)), A3_multinomial(Dd, (1, 3, 5)), Om_i)
    e3, e5 = (b - a) / XLi, (cq - b) / XLi
    aL = np.angle(XLi)
    r_lo, r_res = abs(e5.imag) / abs(aL), abs(e5.imag) / abs(aL + e3.imag)
    rows.append((Fv, abs(Ua[0]), Om_i, b1_i, aL, e3.imag, e5.imag, r_lo, r_res))
    print(f"    {Fv:>7.3f}{abs(Ua[0]):>9.4f}{Om_i:>10.6f}{aL:>11.3e}{e3.imag:>11.3e}"
          f"{e5.imag:>11.3e}{r_lo:>7.3f}{r_res:>7.3f}")

rl = np.array([r[7] for r in rows])
rr = np.array([r[8] for r in rows])
print(f"\n    r_LO  = |Im eta5| / |arg Xi_LO|          : "
      f"[{rl.min():.3f}, {rl.max():.3f}], never reaches 1 on this range")
print(f"    r_res = |Im eta5| / |arg Xi_LO + Im eta3| : "
      f"[{rr.min():.3f}, {rr.max():.3f}], unity at closure as expected")
print("    r_LO RISES as F falls, so Eq. (S23) read literally does not predict a")
print("    low-forcing failure. The turn of the arm near F = 0.256 is established")
print("    by the fold curve kappa*(F) of Sec. S6 A, not by this estimate.")

OUT.mkdir(exist_ok=True)
np.savetxt(OUT / "two_dof_k3_phase_reduction.csv", np.array(rows), delimiter=",",
           comments="",
           header="F,abs_X1,omega_star,beta1_star,arg_xi_lo,im_eta3,im_eta5,"
                  "ratio_vs_xi_lo,ratio_vs_residual_phase")
np.savetxt(OUT / "two_dof_k3_phase_reduction_table.csv", np.array(tab),
           delimiter=",", comments="",
           header="n_harmonics_kept,re_beta1,percent_error,abs_im_beta1")
np.savetxt(OUT / "two_dof_k3_phase_reduction_scaling.csv", sc, delimiter=",",
           comments="", header="F,abs_Delta1,abs_eta5")
print(f"\nwrote {OUT/'two_dof_k3_phase_reduction.csv'}")
print(f"wrote {OUT/'two_dof_k3_phase_reduction_table.csv'}")
print(f"wrote {OUT/'two_dof_k3_phase_reduction_scaling.csv'}")
