"""Analytic two-pathway cancellation condition for the k=3 exact zero.

Derives and evaluates the leading-order third-harmonic cancellation condition

    (L2 + kappa) * beta1 * X1**3  +  kappa_nl * L2 * (X1 - X2)**3  =  0,
    L2 = w2**2 - 9*Om**2 + 2i*z2*(3*Om),

where X1, X2 are the fundamental complex amplitudes. The third harmonic of x1 is
sourced by the two cubic terms (on-site beta1 x1**3 and coupling kappa_nl (x1-x2)**3)
and filtered through the linear 2-DOF dynamic stiffness at 3*Omega. Reproduces the
numbers quoted in the main text and SI Sec. S19:

  * beta1 from the real part at Omega* = 0.2597 : 0.231 (linear fundamental, 3% of 0.2382)
                                                  0.236 (nonlinear fundamental, 1%)
  * the two pathways equal and opposite to four figures (6.103e-3 each)
  * single-harmonic balance cancels 99.6% of the third-harmonic source
  * kappa_nl = 0 -> no nontrivial root; |X3| minimal only near the mode-2
    antiresonance Omega = sqrt((w2**2+kappa)/9) = 0.4298
"""
import csv
import os
import numpy as np
from scipy.optimize import brentq, least_squares

# Baseline oscillator parameters + k=3 working point
W1, W2 = 1.0, 1.25
Z1, Z2 = 0.015, 0.02
KAP, KNL, F = 0.10, -0.30, 0.30
OM_STAR = 0.259728          # full-HB exact-zero frequency
B1_NUM = 0.238154           # full-HB beta1*


def a(j, Om):
    """Fundamental linear diagonal (mode j) at Omega, including coupling kappa."""
    wj, zj = (W1, Z1) if j == 1 else (W2, Z2)
    return wj**2 + KAP - Om**2 + 2j * zj * Om


def L(j, Om):
    """Third-harmonic linear diagonal (mode j) at 3*Omega, without coupling kappa."""
    wj, zj = (W1, Z1) if j == 1 else (W2, Z2)
    return wj**2 - 9 * Om**2 + 2j * zj * (3 * Om)


def linear_fundamental(Om):
    a1, a2 = a(1, Om), a(2, Om)
    det = a1 * a2 - KAP**2
    return F * a2 / det, F * KAP / det


def nonlinear_fundamental(Om, b1):
    """Single-harmonic fundamental with the 3/4 |.|^2 cubic self-terms."""
    a1, a2 = a(1, Om), a(2, Om)

    def res(v):
        X1 = v[0] + 1j * v[1]
        X2 = v[2] + 1j * v[3]
        U = X1 - X2
        f1 = a1 * X1 - KAP * X2 + 0.75 * b1 * abs(X1)**2 * X1 \
            + 0.75 * KNL * abs(U)**2 * U - F
        f2 = a2 * X2 - KAP * X1 - 0.75 * KNL * abs(U)**2 * U
        return [f1.real, f1.imag, f2.real, f2.imag]

    x0 = linear_fundamental(Om)
    s = least_squares(res, [x0[0].real, x0[0].imag, x0[1].real, x0[1].imag],
                      xtol=1e-14, ftol=1e-14)
    return s.x[0] + 1j * s.x[1], s.x[2] + 1j * s.x[3]


def cancellation(Om, b1, fundamental):
    """Return (total, on-site pathway, coupling pathway) of Eq. (cancel)."""
    X1, X2 = fundamental(Om, b1) if fundamental is nonlinear_fundamental \
        else fundamental(Om)
    U = X1 - X2
    on_site = (L(2, Om) + KAP) * b1 * X1**3
    coupling = KNL * L(2, Om) * U**3
    return on_site + coupling, on_site, coupling


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "data", "two_dof_k3_analytic_cancellation.csv")

    # beta1 from the real part of the condition at the true Omega*, linear fundamental
    b1_lin = brentq(lambda b: cancellation(OM_STAR, b, linear_fundamental)[0].real,
                    0.05, 0.60)
    # ... and with a single-harmonic nonlinear fundamental
    b1_nl = brentq(lambda b: cancellation(OM_STAR, b, nonlinear_fundamental)[0].real,
                   0.05, 0.60)

    tot, on_site, coupling = cancellation(OM_STAR, b1_nl, nonlinear_fundamental)
    frac_cancelled = 1.0 - abs(tot) / abs(on_site)

    # kappa_nl = 0 control: mode-2 antiresonance at 3*Omega
    om_mode2 = np.sqrt((W2**2 + KAP) / 9)

    rows = [
        ("beta1_linear_fundamental", b1_lin),
        ("beta1_nonlinear_fundamental", b1_nl),
        ("beta1_full_HB", B1_NUM),
        ("rel_err_linear", abs(b1_lin - B1_NUM) / B1_NUM),
        ("rel_err_nonlinear", abs(b1_nl - B1_NUM) / B1_NUM),
        ("pathway_onsite_abs", abs(on_site)),
        ("pathway_coupling_abs", abs(coupling)),
        ("fraction_cancelled_single_harmonic", frac_cancelled),
        ("residual_abs", abs(tot)),
        ("omega_mode2_antires_knl0", om_mode2),
    ]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        for k, v in rows:
            w.writerow([k, f"{v:.6g}"])

    print(f"beta1 (linear fundamental)    = {b1_lin:.4f}  "
          f"({100*abs(b1_lin-B1_NUM)/B1_NUM:.1f}% from full-HB {B1_NUM})")
    print(f"beta1 (nonlinear fundamental) = {b1_nl:.4f}  "
          f"({100*abs(b1_nl-B1_NUM)/B1_NUM:.1f}%)")
    print(f"pathways |on-site|={abs(on_site):.4e}  |coupling|={abs(coupling):.4e}")
    print(f"single-harmonic cancellation  = {100*frac_cancelled:.2f}% "
          f"(residual {100*abs(tot)/abs(on_site):.2f}%)")
    print(f"kappa_nl=0 mode-2 antiresonance at Omega = {om_mode2:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
