"""Local normal-form coefficient extraction at a detected fold.

The manuscript asserts the local normal form

    A_k^2(Omega; F, lambda) ~ c_0 (Omega - Omega_*)^2
                            + c_1 (F - F_*)
                            + c_2 (lambda - lambda_*)  + O(higher)

at a fold (Omega_*, F_*, lambda_*) of the codim-2 exact-zero manifold. The
codim-2 -> codim-1 shadow argument rests on c_0 > 0 (genuine saddle-node, not
cusp) together with nonzero (c_1, c_2) (the normal form is a transverse
unfolding). This module evaluates A_k^2 at a finite-difference stencil around
(Omega_*, F_*, lambda_*) on the HB manifold, fits the three coefficients, and
reports a cubic-order residual so that the reader can see the normal form is
a verified local geometry rather than an assumed one.

The extraction uses ``solve_harmonic_balance`` for every stencil point, so it
is a full HB evaluation (no linearization). Runtime is dominated by the
per-point HB solves; the default stencil is 5+3+3 = 11 points.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as _replace
from typing import Any

import numpy as np

from .harmonic_balance import (
    harmonic_amplitude_squared_from_coeffs,
    solve_harmonic_balance,
)
from .models import CoupledOscillatorParams


@dataclass
class NormalFormCoefficients:
    """Fitted coefficients of A_k^2 ~ c0*(Om-Om*)^2 + c1*(F-F*) + c2*(lam-lam*).

    Attributes
    ----------
    c0, c1, c2
        Fitted coefficients.
    a_at_fold
        Measured A_k^2 at (Omega_*, F_*, lambda_*). For a true exact-zero fold
        this should be at or below solver precision; a larger value flags that
        either the fold location is imprecise or the fold is not an exact-zero.
    cubic_residual
        Leading-order-3 residual of the Omega-direction fit: ratio of the RMS
        residual after the quadratic fit to |c0| * delta_omega^2. Small values
        (<< 1) certify that the quadratic is the correct local form and the
        fold is a genuine saddle-node, not a cusp.
    stencil
        Metadata on the stencil used (deltas, sample counts).
    """

    c0: float
    c1: float
    c2: float
    a_at_fold: float
    cubic_residual: float
    stencil: dict[str, Any] = field(default_factory=dict)


def _a_squared(
    params: CoupledOscillatorParams,
    *,
    harmonic: int,
    oscillator: int,
    n_harmonics: int,
    n_time_samples: int,
    initial_guess: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    coeffs = solve_harmonic_balance(
        params, n_harmonics=n_harmonics, n_time_samples=n_time_samples,
        initial_guess=initial_guess,
    )
    a2 = harmonic_amplitude_squared_from_coeffs(
        coeffs, harmonic=harmonic, oscillator=oscillator, n_harmonics=n_harmonics
    )
    return float(a2), np.asarray(coeffs, dtype=float)


def fit_fold_normal_form(
    params: CoupledOscillatorParams,
    *,
    fold_omega: float,
    fold_force: float,
    fold_lambda: float,
    unfolding_parameter: str,
    harmonic: int,
    oscillator: int,
    n_harmonics: int,
    n_time_samples: int,
    delta_omega: float = 0.01,
    delta_force: float = 0.002,
    delta_lambda: float = 0.01,
    n_omega_samples: int = 5,
) -> NormalFormCoefficients:
    """Fit the local normal-form coefficients at a detected fold.

    Parameters
    ----------
    params
        Baseline oscillator parameters. ``drive_omega``, ``force``, and the
        unfolding parameter are overridden by the fold location.
    fold_omega, fold_force, fold_lambda
        Refined apex coordinates (typically from ``refine_fold_location``).
    unfolding_parameter
        Name of the parameter in ``CoupledOscillatorParams`` used as lambda.
    harmonic, oscillator, n_harmonics, n_time_samples
        HB evaluation settings.
    delta_omega, delta_force, delta_lambda
        Step sizes for the finite-difference stencils. Choose them small enough
        that the local normal form dominates but large enough that the
        quadratic signal is above solver noise.
    n_omega_samples
        Number of Omega values (odd, >= 3) used for the Omega-direction
        quadratic fit. The samples span [Omega_* - delta_omega, Omega_* +
        delta_omega] uniformly.

    Returns
    -------
    NormalFormCoefficients
    """
    if n_omega_samples < 3 or n_omega_samples % 2 == 0:
        raise ValueError("n_omega_samples must be an odd integer >= 3")
    kwargs = dict(
        harmonic=harmonic, oscillator=oscillator,
        n_harmonics=n_harmonics, n_time_samples=n_time_samples,
    )
    base = _replace(
        params, drive_omega=float(fold_omega), force=float(fold_force),
        **{unfolding_parameter: float(fold_lambda)},
    )
    a2_fold, coeffs_fold = _a_squared(base, **kwargs)

    # Omega direction: quadratic fit of A_k^2(Omega) at fixed (F_*, lambda_*).
    omegas = np.linspace(
        fold_omega - delta_omega, fold_omega + delta_omega, n_omega_samples
    )
    a2_vs_omega = np.empty(n_omega_samples, dtype=float)
    guess = coeffs_fold
    for i, om in enumerate(omegas):
        p_i = _replace(base, drive_omega=float(om))
        val, guess = _a_squared(p_i, initial_guess=guess, **kwargs)
        a2_vs_omega[i] = val

    # Least-squares quadratic A^2 ~ c0 (Om - Om*)^2 + d1 (Om - Om*) + d0.
    x = omegas - fold_omega
    A = np.vstack([x * x, x, np.ones_like(x)]).T
    (c0, _d1, _d0), *_ = np.linalg.lstsq(A, a2_vs_omega, rcond=None)
    fit_omega = A @ np.array([c0, _d1, _d0])
    rms_resid = float(np.sqrt(np.mean((a2_vs_omega - fit_omega) ** 2)))
    cubic_norm = abs(float(c0)) * (float(delta_omega) ** 2) + 1e-30
    cubic_residual = rms_resid / cubic_norm

    # F direction: central difference at (Omega_*, lambda_*).
    p_fp = _replace(base, force=float(fold_force + delta_force))
    p_fm = _replace(base, force=float(fold_force - delta_force))
    a2_fp, _ = _a_squared(p_fp, initial_guess=coeffs_fold, **kwargs)
    a2_fm, _ = _a_squared(p_fm, initial_guess=coeffs_fold, **kwargs)
    c1 = (a2_fp - a2_fm) / (2.0 * delta_force)

    # lambda direction: central difference at (Omega_*, F_*).
    p_lp = _replace(base, **{unfolding_parameter: float(fold_lambda + delta_lambda)})
    p_lm = _replace(base, **{unfolding_parameter: float(fold_lambda - delta_lambda)})
    a2_lp, _ = _a_squared(p_lp, initial_guess=coeffs_fold, **kwargs)
    a2_lm, _ = _a_squared(p_lm, initial_guess=coeffs_fold, **kwargs)
    c2 = (a2_lp - a2_lm) / (2.0 * delta_lambda)

    return NormalFormCoefficients(
        c0=float(c0),
        c1=float(c1),
        c2=float(c2),
        a_at_fold=float(a2_fold),
        cubic_residual=float(cubic_residual),
        stencil={
            "delta_omega": float(delta_omega),
            "delta_force": float(delta_force),
            "delta_lambda": float(delta_lambda),
            "n_omega_samples": int(n_omega_samples),
            "unfolding_parameter": unfolding_parameter,
        },
    )
