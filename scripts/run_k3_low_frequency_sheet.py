"""What the "low-frequency sheet" in the wide seed hunt actually is.

run_k3_multimethod_verification.py bounds its augmented solve inside the search
window because, unbounded, it escapes to a low-frequency sheet that "otherwise
satisfies the constraints". Calling that sheet trivial and moving on is not an
analysis, so this driver characterises it.

The sheet is the Omega -> 0 boundary of the two-quadrature condition. Im Xi_LO
is odd in Omega, so Im Xi_LO = c0*Omega + O(Omega^3) with c0 > 0
(run_k3_leading_order_scan.py). The phase obstruction that a real beta_1 cannot
close therefore shrinks proportionally to Omega, and a single real control can
meet both quadratures to O(Omega). That is a degeneracy of the condition at the
boundary, not a family of exact zeros.

Test: at fixed forcing, minimise |X_3| over beta_1 at a ladder of decreasing
Omega. If the sheet were a family of exact zeros the minimum would hit the
solver floor; if it is the boundary degeneracy the minimum stays nonzero and
falls like Omega.

Run:  PYTHONPATH=src JAX_ENABLE_X64=1 uv run python scripts/run_k3_low_frequency_sheet.py
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
from pathlib import Path

import numpy as np
from jax import config

config.update("jax_enable_x64", True)
from scipy.optimize import minimize_scalar

from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import solve_harmonic_balance, coefficient_index

W1, W2, Z1, Z2 = 1.0, 1.25, 0.015, 0.020
KA, KNL, F0 = 0.10, -0.30, 0.30
NH = 7

_D = Path(os.path.dirname(os.path.abspath(__file__)))
DATA = _D / ".." / "data"


def xi_lo(Om):
    L2 = W2 ** 2 - (3 * Om) ** 2 + 2j * Z2 * (3 * Om)
    Z3 = L2 + KA
    Z1w = W2 ** 2 - Om ** 2 + 2j * Z2 * Om + KA
    return -KNL * L2 * (1 - KA / Z1w) ** 3 / Z3


_cache = {"g": None}


def amp3(Om, b1):
    p = CoupledOscillatorParams(omega1=W1, omega2=W2, zeta1=Z1, zeta2=Z2,
                                alpha1=0.0, alpha2=0.0, beta1=float(b1), beta2=0.0,
                                kappa=KA, kappa_nl=KNL, force=F0, drive_omega=Om)
    c = solve_harmonic_balance(p, n_harmonics=NH, n_time_samples=1024, tol=1e-13,
                               initial_guess=_cache["g"])
    _cache["g"] = c
    i = lambda h, cp: coefficient_index(oscillator=1, harmonic=h, component=cp, n_harmonics=NH)
    return abs(complex(c[i(3, "cos")], -c[i(3, "sin")])), abs(complex(c[i(1, "cos")], -c[i(1, "sin")]))


def main():
    print("=" * 88)
    print('THE "LOW-FREQUENCY SHEET": THE Omega -> 0 BOUNDARY, NOT A FAMILY OF ZEROS')
    print("=" * 88)
    print(f"F = {F0} fixed, beta_1 free, N_H = {NH}\n")
    print(f"{'Omega':>8}{'Re Xi_LO':>12}{'beta_1 at min':>15}{'min |X_3|':>13}"
          f"{'min|X3| / Omega':>17}{'|X_1,1|':>10}")
    rows = []
    for Om in (0.30, 0.20, 0.12, 0.08, 0.05, 0.03, 0.02):
        b0 = xi_lo(Om).real
        _cache["g"] = None
        r = minimize_scalar(lambda b: amp3(Om, b)[0], bracket=None,
                            bounds=(b0 - 0.03, b0 + 0.03), method="bounded",
                            options={"xatol": 1e-8})
        v, a1 = amp3(Om, r.x)
        rows.append([Om, b0, r.x, v, a1])
        print(f"{Om:8.3f}{b0:12.6f}{r.x:15.6f}{v:13.3e}{v / Om:17.3e}{a1:10.5f}")

    a = np.array(rows)
    tail = a[a[:, 0] <= 0.08]
    sl_all = np.polyfit(np.log(a[:, 0]), np.log(a[:, 3]), 1)[0]
    sl_tail = np.polyfit(np.log(tail[:, 0]), np.log(tail[:, 3]), 1)[0]
    ratio = tail[:, 3] / tail[:, 0]
    print(f"\n  exponent of min|X_3| vs Omega: {sl_all:.2f} over the whole ladder,")
    print(f"  {sl_tail:.2f} over the asymptotic tail Omega <= 0.08. The c0*Omega law is")
    print(f"  reached only asymptotically: min|X_3|/Omega settles to")
    print(f"  {ratio.min():.2e}..{ratio.max():.2e} there, a spread of "
          f"{100 * (ratio.max() - ratio.min()) / ratio.mean():.0f}%.")
    assert a[:, 3].min() > 1e-9, "a genuine zero was reached: the sheet is not a degeneracy"
    print("  The minimum never reaches the solver floor: at every frequency tested the two")
    print("  quadratures remain unmet. The sheet is the boundary degeneracy of the")
    print("  cancellation condition, where the phase obstruction vanishes with the")
    print("  frequency, not a locus of exact zeros.")
    np.savetxt(DATA / "two_dof_k3_low_frequency_sheet.csv", a, delimiter=",",
               header="omega,Re_xi_lo,beta1_at_min,min_abs_X3,abs_X1", comments="")
    print(f"\nwrote {DATA / 'two_dof_k3_low_frequency_sheet.csv'}")


if __name__ == "__main__":
    main()
