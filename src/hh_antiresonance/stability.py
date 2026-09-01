"""Floquet-multiplier stability analysis of harmonic-balance periodic orbits.

For a harmonic-balance solution, the full-ODE steady-state orbit $x^*(t)$ is
periodic with the drive period $T = 2\\pi/\\Omega$. Linearizing the 4D
state-space system $\\dot y = f(y,t)$ around $y^*(t)$ gives the variational
equation

    $\\dot\\Phi(t) = J(t)\\,\\Phi(t), \\quad \\Phi(0) = I_4,$

where $J(t)$ is the Jacobian of $f$ along the orbit. The *monodromy matrix*
$M = \\Phi(T)$ has four Floquet multipliers; the orbit is asymptotically
stable iff all multipliers lie strictly inside the unit circle.

We reconstruct $y^*(t)$ directly from the HB Fourier coefficients (no
additional time integration), integrate the 16-component matrix ODE with the
same explicit scheme used in the full-ODE cross-check, and return both the
multipliers and a compact classification summary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

from .harmonic_balance import unpack_coefficients
from .models import CoupledOscillatorParams


@dataclass(frozen=True)
class FloquetReport:
    """Result of a monodromy-based stability assessment at one HB solution."""
    multipliers: np.ndarray          # shape (4,), complex
    spectral_radius: float
    largest_real_log: float          # max Re(ln mu_i) / T  (Lyapunov-like)
    determinant: complex
    trace_predicted_log_det: float   # -2*(zeta1+zeta2)*T; det(M) should equal exp of this
    is_stable: bool
    solver_message: str


def _orbit_state(t: float, coeffs: np.ndarray, omega: float, n_harmonics: int) -> tuple[float, float]:
    """Return (x1, x2) at time t from the HB Fourier coefficients."""
    x1_c, x2_c = unpack_coefficients(coeffs, n_harmonics)
    x1 = 0.0
    x2 = 0.0
    for m in range(1, n_harmonics + 1):
        w = m * omega
        c = np.cos(w * t)
        s = np.sin(w * t)
        a1, b1 = x1_c[m - 1]
        a2, b2 = x2_c[m - 1]
        x1 += a1 * c + b1 * s
        x2 += a2 * c + b2 * s
    return x1, x2


def _variational_jacobian(
    x1: float, x2: float, params: CoupledOscillatorParams
) -> np.ndarray:
    """Jacobian of the 4D state-space RHS at (x1, x2) on the orbit."""
    J = np.zeros((4, 4), dtype=float)
    # y = (x1, v1, x2, v2);  f = (v1, dv1, v2, dv2).
    # Cubic coupling kappa_nl*(x1-x2)^3 contributes a state-dependent stiffness
    # 3*kappa_nl*(x1-x2)^2 on the relative coordinate.
    dx = x1 - x2
    k_nl_tangent = 3.0 * params.kappa_nl * dx ** 2
    J[0, 1] = 1.0
    J[1, 0] = -(
        params.omega1 ** 2
        + 2.0 * params.alpha1 * x1
        + 3.0 * params.beta1 * x1 ** 2
        + params.kappa
        + k_nl_tangent
    )
    J[1, 1] = -2.0 * params.zeta1
    J[1, 2] = params.kappa + k_nl_tangent
    J[2, 3] = 1.0
    J[3, 0] = params.kappa + k_nl_tangent
    J[3, 2] = -(
        params.omega2 ** 2
        + 2.0 * params.alpha2 * x2
        + 3.0 * params.beta2 * x2 ** 2
        + params.kappa
        + k_nl_tangent
    )
    J[3, 3] = -2.0 * params.zeta2
    return J


def floquet_multipliers_from_coeffs(
    params: CoupledOscillatorParams,
    coeffs: np.ndarray,
    *,
    n_harmonics: int,
    ode_method: Literal["RK45", "DOP853", "Radau", "BDF", "LSODA"] = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> FloquetReport:
    """Integrate the variational equation over one drive period; report multipliers."""
    if params.drive_omega <= 0.0:
        raise ValueError("drive_omega must be positive")
    omega = float(params.drive_omega)
    T = 2.0 * np.pi / omega
    coeffs = np.asarray(coeffs, dtype=float)

    def rhs(t: float, Phi_flat: np.ndarray) -> np.ndarray:
        x1, x2 = _orbit_state(t, coeffs, omega, n_harmonics)
        J = _variational_jacobian(x1, x2, params)
        Phi = Phi_flat.reshape(4, 4)
        return (J @ Phi).reshape(-1)

    Phi0 = np.eye(4, dtype=float).reshape(-1)
    sol = solve_ivp(
        rhs,
        t_span=(0.0, T),
        y0=Phi0,
        method=ode_method,
        rtol=rtol,
        atol=atol,
        dense_output=False,
        max_step=T / 32.0,
        t_eval=[T],
    )
    if not sol.success:
        raise RuntimeError(f"Variational integration failed: {sol.message}")

    M = sol.y[:, -1].reshape(4, 4)
    multipliers = np.linalg.eigvals(M)
    spectral_radius = float(np.max(np.abs(multipliers)))
    # ln|mu| / T is the Lyapunov-like rate. We report the largest.
    with np.errstate(divide="ignore"):
        largest_real_log = float(np.max(np.log(np.abs(multipliers))) / T)
    det_M = complex(np.linalg.det(M))
    # Abel--Liouville: det(M) = exp(int_0^T trace(J(t)) dt) = exp(-2(zeta1+zeta2) T).
    predicted = -2.0 * (params.zeta1 + params.zeta2) * T
    is_stable = spectral_radius < 1.0

    return FloquetReport(
        multipliers=multipliers,
        spectral_radius=spectral_radius,
        largest_real_log=largest_real_log,
        determinant=det_M,
        trace_predicted_log_det=float(predicted),
        is_stable=bool(is_stable),
        solver_message=sol.message,
    )


def assess_hb_stability(
    params: CoupledOscillatorParams,
    *,
    n_harmonics: int = 3,
    n_time_samples: int = 256,
    hb_initial_guess: np.ndarray | None = None,
    stability_kwargs: dict | None = None,
) -> dict[str, float | bool | str]:
    """Solve HB at the given parameters and classify the orbit's Floquet stability."""
    from .harmonic_balance import solve_harmonic_balance

    stability_kwargs = dict(stability_kwargs or {})
    try:
        coeffs = solve_harmonic_balance(
            params,
            n_harmonics=n_harmonics,
            initial_guess=hb_initial_guess,
            n_time_samples=n_time_samples,
        )
        hb_status = "solved"
    except Exception as exc:
        return {
            "hb_status": f"hb_failed:{type(exc).__name__}",
            "spectral_radius": float("nan"),
            "largest_real_log": float("nan"),
            "is_stable": False,
            "det_abs_error": float("nan"),
            "stability_status": "skipped",
        }

    try:
        report = floquet_multipliers_from_coeffs(
            params, coeffs, n_harmonics=n_harmonics, **stability_kwargs
        )
        det_abs_error = float(abs(abs(report.determinant) - np.exp(report.trace_predicted_log_det)))
        return {
            "hb_status": hb_status,
            "spectral_radius": report.spectral_radius,
            "largest_real_log": report.largest_real_log,
            "is_stable": report.is_stable,
            "det_abs_error": det_abs_error,
            "stability_status": "assessed",
        }
    except Exception as exc:
        return {
            "hb_status": hb_status,
            "spectral_radius": float("nan"),
            "largest_real_log": float("nan"),
            "is_stable": False,
            "det_abs_error": float("nan"),
            "stability_status": f"floquet_failed:{type(exc).__name__}",
        }
