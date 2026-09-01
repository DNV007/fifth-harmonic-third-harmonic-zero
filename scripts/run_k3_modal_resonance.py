"""Modal structure at the exact zero: the fifth-harmonic propagator enhancement.

At the baseline parameters the two coupled linear modes sit at

    omega_- = 1.040553,   omega_+ = 1.296051     (omega_+/omega_- = 1.24554)

because omega_2/omega_1 = 5/4 was chosen. The reported root Omega* = 0.259728
therefore places

    5*Omega* = 1.298640   within 0.20% of omega_+     (odd -> ACTIVE)
    4*Omega* = 1.038912   within 0.16% of omega_-     (even -> parity-suppressed)

Only the odd multiple is dynamically live: with alpha_j = 0 the orbit is
half-period antisymmetric and every even harmonic vanishes to ~1e-16.

This driver quantifies the consequence. The relevant quantity is NOT the
frequency ratio but the propagator enhancement

    eta(Omega) = |det D_3(Omega)| / |det D_5(Omega)|,

the responsiveness of the fifth-harmonic dynamic stiffness relative to the
third at the same operating point. A5 itself is small because its SOURCE is
small (it is generated at 3Om+Om+Om through the absorber's third harmonic,
the driven-mass third harmonic being zero by construction); small amplitude is
not evidence of weak enhancement.

Reported:
  (1) linear modes, harmonic placement, and eta at the baseline root
  (2) eta and the 5:1 detuning at every perturbed root reported in the SM,
      re-solved here rather than read from the text
  (3) the Lorentzian half-width of the omega_+ resonance, which sets how many
      linewidths each test actually moved

Run:  uv run python scripts/run_k3_modal_resonance.py
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
OM_ARM_B = 0.2515209
N, NT = 7, 512
_D = Path(os.path.dirname(os.path.abspath(__file__)))
OUT = _D / ".." / "data"

nn = np.arange(NT)
PH = 2 * np.pi * np.outer(np.arange(1, N + 1), nn) / NT
COS, SIN = np.cos(PH), np.sin(PH)
MVEC = np.arange(1, N + 1)[:, None]
IC, IS = 2 * (3 - 1), 2 * (3 - 1) + 1


def linear_modes(w2=W2, ka=KA, mu=1.0, w1=W1):
    """Squared-frequency eigenvalues of the coupled linear stiffness."""
    K = np.array([[w1 ** 2 + ka, -ka], [-ka / mu, w2 ** 2 + ka / mu]])
    lam = np.sort(np.linalg.eigvals(K).real)
    return np.sqrt(lam)


def detD(k, Om, w2=W2, ka=KA, mu=1.0, w1=W1):
    kO = k * Om
    return ((w1 ** 2 + ka - kO ** 2 + 2j * Z1 * kO)
            * (w2 ** 2 + ka / mu - kO ** 2 + 2j * Z2 * kO) - ka * ka / mu)


def Dmat(k, Om, w2=W2, ka=KA, mu=1.0, w1=W1):
    kO = k * Om
    return np.array([[w1 ** 2 + ka - kO ** 2 + 2j * Z1 * kO, -ka],
                     [-ka / mu, w2 ** 2 + ka / mu - kO ** 2 + 2j * Z2 * kO]])


def eta(Om, **kw):
    """Determinant ratio. Proximity to singularity ONLY -- not a gain."""
    return abs(detD(3, Om, **kw)) / abs(detD(5, Om, **kw))


def sigma_ratio(Om, **kw):
    """sigma_min(D_3)/sigma_min(D_5): ratio of operator norms ||D^-1||_2."""
    s3 = np.linalg.svd(Dmat(3, Om, **kw), compute_uv=False)
    s5 = np.linalg.svd(Dmat(5, Om, **kw), compute_uv=False)
    return float(s3[-1] / s5[-1])


def gain_rel(Om, mu=1.0, **kw):
    """Projected gain along the live channel: coupling source in, d = x1-x2 out.

    A unit coupling source enters eq.1 with +1 and eq.2 with -1/mu; the readout
    r = (1,-1) is the relative coordinate the coupling cubic acts on.
    """
    S = np.array([1.0, -1.0 / mu])
    r = np.array([1.0, -1.0])
    g3 = abs(r @ np.linalg.solve(Dmat(3, Om, mu=mu, **kw), S))
    g5 = abs(r @ np.linalg.solve(Dmat(5, Om, mu=mu, **kw), S))
    return float(g5 / g3)


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


def hb_res(c, Om, b1, F=F0, mu=1.0, w2=W2, k5=0.0):
    q1, v1, ac1, q2, v2, ac2 = recon(c, Om)
    d = q1 - q2
    cpl = KA * d + KNL * d ** 3 + k5 * d ** 5
    r1 = ac1 + 2 * Z1 * v1 + W1 ** 2 * q1 + b1 * q1 ** 3 + cpl - F * COS[0]
    r2 = ac2 + 2 * Z2 * v2 + w2 ** 2 * q2 - cpl / mu
    out = np.empty(4 * N)
    for m in range(N):
        out[4 * m:4 * m + 4] = [(2 / NT) * (r1 @ COS[m]), (2 / NT) * (r1 @ SIN[m]),
                                (2 / NT) * (r2 @ COS[m]), (2 / NT) * (r2 @ SIN[m])]
    return out


def solve_hb(Om, b1, F=F0, mu=1.0, w2=W2, k5=0.0, guess=None):
    z0 = np.zeros(4 * N) if guess is None else guess.copy()
    if guess is None:
        z0[0] = F / (W1 ** 2 - Om ** 2)
    else:
        s = root(lambda c: hb_res(c, Om, b1, F, mu, w2, k5), z0, method="hybr", tol=1e-13)
        if s.success:
            return s.x
    return least_squares(lambda c: hb_res(c, Om, b1, F, mu, w2, k5), z0,
                         xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=6000).x


def solve_zero(seed, F=F0, mu=1.0, w2=W2, k5=0.0):
    def aug(z):
        c = z[:4 * N]
        return np.concatenate([hb_res(c, float(z[-2]), float(z[-1]), F, mu, w2, k5),
                               [c[IC], c[IS]]])
    s = least_squares(aug, seed, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=9000)
    return s.x


def amps(c):
    """(|X1|, |X3|, A5) of the driven mass."""
    def a(k):
        i = 2 * (k - 1)
        return float(np.hypot(c[i], c[i + 1]))
    return a(1), a(3), a(5)


def upper_pole(w2=W2, ka=KA, mu=1.0, w1=W1):
    """Complex pole of the upper mode: returns (damped freq, half-width alpha)."""
    K = np.array([[w1 ** 2 + ka, -ka], [-ka / mu, w2 ** 2 + ka / mu]])
    C = np.diag([2 * Z1, 2 * Z2])
    A = np.block([[np.zeros((2, 2)), np.eye(2)], [-K, -C]])
    ev = [s for s in np.linalg.eigvals(A) if s.imag > 0]
    s = max(ev, key=lambda z: z.imag)
    return float(s.imag), float(-s.real)


print("=" * 92)
print("MODAL STRUCTURE AND THE FIFTH-HARMONIC PROPAGATOR AT THE EXACT ZERO")
print("=" * 92)

wm, wp = linear_modes()
wd, alpha = upper_pole()
print(f"\n(1) baseline linear modes  (omega_2/omega_1 = {W2/W1:.2f} = 5/4)")
print(f"    omega_- = {wm:.6f}   omega_+ = {wp:.6f}   omega_+/omega_- = {wp/wm:.6f}")
print(f"    upper complex pole: damped freq {wd:.6f}, half-width alpha = {alpha:.6f}")
print(f"    (alpha differs from zeta_2 = {Z2} by {abs(alpha/Z2-1)*100:.2f}%; "
      f"'alpha ~ zeta_2' is an approximation, not an identity)")
print(f"\n    at Omega* = {OM_PROD:.6f}:")
print(f"      4*Omega* = {4*OM_PROD:.6f}  vs omega_- : {(4*OM_PROD/wm-1)*100:+.3f}%   "
      f"[EVEN -> parity-suppressed]")
print(f"      5*Omega* = {5*OM_PROD:.6f}  vs omega_+ : {(5*OM_PROD/wp-1)*100:+.3f}%   "
      f"[ODD  -> active, generated at 3*Om + Om + Om = 5*Om]")

print(f"\n(2) PROXIMITY TO SINGULARITY (determinants -- NOT gains)")
for k in (1, 3, 4, 5, 7):
    print(f"      |det D_{k}| = {abs(detD(k, OM_PROD)):.6f}")
print(f"      |det D_3|/|det D_5| = {eta(OM_PROD):.2f}   "
      f"|det D_3|/|det D_4| = {abs(detD(3,OM_PROD))/abs(detD(4,OM_PROD)):.2f}")
print("      A determinant ratio is the product of both singular values; it bounds")
print("      nothing about ||D^-1|| on its own. The gains below are the real measure.")

print(f"\n(3) PROPAGATOR GAIN at Omega*")
for k in (3, 5):
    sv = np.linalg.svd(Dmat(k, OM_PROD), compute_uv=False)
    print(f"      k={k}: sigma = ({sv[0]:.5f}, {sv[1]:.5f}),  ||D^-1||_2 = {1/sv[1]:.4f}")
print(f"      operator-norm ratio  sigma_min(D_3)/sigma_min(D_5) = {sigma_ratio(OM_PROD):.3f}")
S = np.array([1.0, -1.0])
for lbl, r in (("driven mass   c = (1, 0)", np.array([1.0, 0.0])),
               ("relative      r = (1,-1)", np.array([1.0, -1.0]))):
    g3 = abs(r @ np.linalg.solve(Dmat(3, OM_PROD), S))
    g5 = abs(r @ np.linalg.solve(Dmat(5, OM_PROD), S))
    print(f"      projected |r^H D^-1 S|, {lbl}:  k=3 {g3:.4f}  k=5 {g5:.4f}  "
          f"ratio {g5/g3:.3f}")
print("      -> the enhancement lives in the RELATIVE coordinate, which is exactly")
print("         what the coupling cubic drives and reads. The driven-mass projection")
print("         is weakly enhanced, which is why A5/|X1| looks small.")

print(f"\n(4) resonance width from the complex pole: alpha = {alpha:.6f} "
      f"({alpha/wp*100:.2f}% of omega_+)")
print(f"    enhancement relative to peak = alpha/sqrt(delta^2+alpha^2):")
for dl in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
    d = dl * alpha
    print(f"      {dl:4.1f} linewidths: {alpha/np.hypot(d,alpha):.3f} of peak")

print(f"\n(5) detuning and gain at every root, RE-SOLVED here")
print(f"    {'case':<22}{'Omega*':>11}{'w_+':>10}{'dev%':>8}"
      f"{'lw':>7}{'sig':>8}{'gain_d':>9}{'A5/|X1|':>10}{'|X3|':>10}")

rows = []


def report(label, z, mu=1.0, w2=W2):
    Om, b1 = float(z[-2]), float(z[-1])
    c = z[:4 * N]
    X1, X3v, A5 = amps(c)
    wmi, wpi = linear_modes(w2=w2, mu=mu)
    _, al = upper_pole(w2=w2, mu=mu)
    dev = (Om / (wpi / 5) - 1) * 100
    lw = abs(5 * Om - wpi) / al
    sr = sigma_ratio(Om, w2=w2, mu=mu)
    gr = gain_rel(Om, w2=w2, mu=mu)
    print(f"    {label:<22}{Om:>11.7f}{wpi:>10.6f}{dev:>+8.2f}"
          f"{lw:>7.2f}{sr:>8.2f}{gr:>9.2f}{A5/X1:>10.2e}{X3v:>10.1e}")
    rows.append((Om, b1, wmi, wpi, dev, lw, sr, gr, A5 / X1, X3v))
    return Om, b1


zA = solve_zero(np.concatenate([solve_hb(OM_PROD, B1_PROD), [OM_PROD, B1_PROD]]))
report("baseline (arm A)", zA)
zB = solve_zero(np.concatenate([solve_hb(OM_ARM_B, B1_PROD), [OM_ARM_B, B1_PROD]]))
report("baseline (arm B)", zB)

# continuation in small steps so we stay on arm A
z = zA
for w2v in np.arange(1.25, 1.3501, 0.0125):
    z = solve_zero(z, w2=float(w2v))
    if abs(w2v - round(w2v, 2)) < 1e-9 and round(w2v, 2) in (1.30, 1.35):
        report(f"omega_2 = {w2v:.3f}", z, w2=float(w2v))
z = zA
for w2v in np.arange(1.25, 1.0999, -0.0125):
    z = solve_zero(z, w2=float(w2v))
    if round(w2v, 4) in (1.175, 1.1):
        report(f"omega_2 = {w2v:.3f}", z, w2=float(w2v))
z = zA
for muv in np.arange(1.0, 0.399, -0.05):
    z = solve_zero(z, mu=float(muv))
    if round(muv, 2) in (0.70, 0.40):
        report(f"mu = {muv:.2f}", z, mu=float(muv))
z = zA
for muv in np.arange(1.0, 2.001, 0.05):
    z = solve_zero(z, mu=float(muv))
    if round(muv, 2) == 2.00:
        report(f"mu = {muv:.2f}", z, mu=float(muv))

print("\n    dev%   = (Omega* - omega_+/5)/(omega_+/5)")
print("    lw     = |5*Omega* - omega_+| / alpha, alpha from the complex pole")
print("    sig    = sigma_min(D_3)/sigma_min(D_5), operator-norm ratio")
print("    gain_d = |r^H D_5^-1 S| / |r^H D_3^-1 S| in the relative coordinate")

sr_all = np.array([r[6] for r in rows]); gr_all = np.array([r[7] for r in rows])
print(f"\n    sigma ratio over ALL roots: [{sr_all.min():.2f}, {sr_all.max():.2f}]")
print(f"    relative-coordinate gain  : [{gr_all.min():.2f}, {gr_all.max():.2f}]")

OUT.mkdir(exist_ok=True)
np.savetxt(OUT / "two_dof_k3_modal_resonance.csv", np.array(rows), delimiter=",",
           comments="",
           header="omega_star,beta1_star,omega_minus,omega_plus,dev_percent,"
                  "linewidths,sigma_ratio,gain_relative,A5_over_X1,abs_X3")
print(f"\nwrote {OUT/'two_dof_k3_modal_resonance.csv'}")
