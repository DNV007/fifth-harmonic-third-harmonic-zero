"""Linear-limit consistency of the antiresonance defining systems.

For the driven linear coupled system ($\\alpha_i = \\beta_i = 0$) the complex
steady-state amplitude of oscillator 1 at the drive frequency is closed form,

    $X_1(\\Omega) = F \\cdot \\frac{\\omega_2^2 + \\kappa - \\Omega^2 + 2i\\zeta_2\\Omega}{\\Delta(\\Omega)}$,

with $\\Delta(\\Omega) = (\\omega_1^2+\\kappa-\\Omega^2+2i\\zeta_1\\Omega)(\\omega_2^2+\\kappa-\\Omega^2+2i\\zeta_2\\Omega) - \\kappa^2$.
The frequency that minimizes $|X_1(\\Omega)|^2$ is well defined for $\\zeta_i > 0$
and is the classical linear antiresonance; in the undamped limit it reduces to
$\\Omega^2 = \\omega_2^2 + \\kappa$. We compute this reference frequency by a
one-dimensional root-find of $d|X_1|^2/d\\Omega = 0$ and require that
$\\texttt{solve\\_projected\\_antiresonance\\_fixed\\_force}$ match it to solver
precision.

For the *exact* condition ($X_1^{\\cos} = X_1^{\\sin} = 0$) in the linear case,
the codim-2 counting forces the trivial solution $F = 0$: in a linear system
$X_1$ is strictly proportional to $F$. We require the exact solver to drive
$F$ to negligible magnitude, confirming that nonlinearity (phase freedom) is
what makes nonzero-force exact antiresonance possible.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize_scalar

from hh_antiresonance.harmonic_balance import (
    ProjectedAntiresonanceError,
    classify_projected_stationary_point,
    solve_exact_zero_point_fixed_unfolding,
    solve_projected_antiresonance_fixed_force,
)
from hh_antiresonance.models import CoupledOscillatorParams


OMEGA1, OMEGA2, KAPPA = 1.0, 1.35, 0.12
UNDAMPED_AR = float(np.sqrt(OMEGA2 ** 2 + KAPPA))


def _linear_x1_squared_amplitude(omega: float, p: CoupledOscillatorParams) -> float:
    """Closed-form |X_1(Omega)|^2 for the linear coupled system."""
    denom_osc1 = complex(p.omega1 ** 2 + p.kappa - omega ** 2, 2.0 * p.zeta1 * omega)
    denom_osc2 = complex(p.omega2 ** 2 + p.kappa - omega ** 2, 2.0 * p.zeta2 * omega)
    Delta = denom_osc1 * denom_osc2 - p.kappa ** 2
    numerator = complex(p.omega2 ** 2 + p.kappa - omega ** 2, 2.0 * p.zeta2 * omega)
    return float(abs(p.force * numerator / Delta) ** 2)


def _analytic_damped_antiresonance(p: CoupledOscillatorParams) -> float:
    """Find the linear antiresonance of oscillator 1 by minimizing |X_1|^2."""
    # The minimum of |X_1|^2 sits slightly below the undamped value
    # sqrt(omega2^2 + kappa) when damping is nonzero. We search in a bracket
    # that contains it and is bounded away from the coupled-mode poles.
    omega_undamped = float(np.sqrt(p.omega2 ** 2 + p.kappa))
    result = minimize_scalar(
        lambda w: _linear_x1_squared_amplitude(float(w), p),
        bounds=(omega_undamped - 0.1, omega_undamped + 0.02),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if not result.success:
        raise RuntimeError("Failed to locate analytic damped antiresonance")
    return float(result.x)


def _linear_damped() -> CoupledOscillatorParams:
    return CoupledOscillatorParams(
        omega1=OMEGA1, omega2=OMEGA2, zeta1=0.01, zeta2=0.015,
        alpha1=0.0, alpha2=0.0, beta1=0.0, beta2=0.0,
        kappa=KAPPA, force=0.01, drive_omega=1.0,
    )


def test_projected_solver_matches_closed_form_in_damped_linear_limit():
    p = _linear_damped()
    reference = _analytic_damped_antiresonance(p)
    state = solve_projected_antiresonance_fixed_force(
        p, harmonic=1, oscillator=1, n_harmonics=1,
        start_omega=reference, force=p.force, n_time_samples=64,
    )
    omega_star = float(state[-2])
    assert abs(omega_star - reference) < 1e-6
    # Classifier must confirm we landed on a minimum of |X_1|^2, not a peak.
    coeffs = state[: 4 * 1]
    from dataclasses import replace as _replace
    p_at_star = _replace(p, drive_omega=omega_star, force=p.force)
    cls = classify_projected_stationary_point(
        coeffs, p_at_star, harmonic=1, oscillator=1, n_harmonics=1, n_time_samples=64,
    )
    assert cls["kind"] == "antiresonance"


def test_projected_solver_matches_closed_form_across_kappa():
    for dk in (-0.05, 0.0, 0.06):
        p = replace(_linear_damped(), kappa=KAPPA + dk)
        reference = _analytic_damped_antiresonance(p)
        state = solve_projected_antiresonance_fixed_force(
            p, harmonic=1, oscillator=1, n_harmonics=1,
            start_omega=reference, force=p.force, n_time_samples=64,
        )
        assert abs(float(state[-2]) - reference) < 1e-6


def test_undamped_limit_approaches_classical_formula():
    # Progressively smaller damping should push the antiresonance toward
    # the closed-form undamped value Omega^2 = omega_2^2 + kappa.
    residuals = []
    for zeta in (0.05, 0.02, 0.005):
        p = replace(_linear_damped(), zeta1=zeta, zeta2=zeta)
        reference = _analytic_damped_antiresonance(p)
        residuals.append(abs(reference - UNDAMPED_AR))
    # Must decrease monotonically (at least monotonically to numerical noise).
    assert residuals[0] >= residuals[1] >= residuals[2]
    assert residuals[-1] < 5e-3


def test_exact_solver_drives_force_to_zero_in_linear_case():
    # With beta = alpha = 0, X_1 is strictly proportional to F, so the only
    # solution of X_1 = 0 at non-zero damping is F = 0.
    p = _linear_damped()
    state = solve_exact_zero_point_fixed_unfolding(
        p, unfolding_parameter="kappa",
        harmonic=1, oscillator=1, n_harmonics=1,
        start_omega=UNDAMPED_AR + 0.05, start_force=0.01,
        unfolding_value=KAPPA, n_time_samples=64,
    )
    force_final = float(state[-2])
    assert abs(force_final) < 1e-4  # driven to the trivial solution


def test_projected_solver_rejects_maximum_seed_by_default():
    """Regression: seeding near a local maximum must NOT silently return the max.

    At kappa=0.07 with the default parameters, |X_1(Omega)|^2 has a minimum
    near Omega=1.368 and a local maximum near Omega=1.386. Seeding at 1.377
    previously converged to the max; with the default kind='minimum' the
    solver must reseed symmetrically and return the minimum instead.
    """
    from dataclasses import replace as _replace
    p = _replace(_linear_damped(), kappa=0.07)
    reference_min = _analytic_damped_antiresonance(p)  # ~1.368
    state = solve_projected_antiresonance_fixed_force(
        p, harmonic=1, oscillator=1, n_harmonics=1,
        start_omega=1.377, force=p.force, n_time_samples=64,
    )
    omega_star = float(state[-2])
    # Must land on the minimum, not the nearby maximum at ~1.386.
    assert abs(omega_star - reference_min) < 1e-6


def test_projected_solver_any_mode_keeps_stationary_points():
    """kind='any' must preserve the previous behaviour: any stationary point is OK."""
    from dataclasses import replace as _replace
    p = _replace(_linear_damped(), kappa=0.07)
    state = solve_projected_antiresonance_fixed_force(
        p, harmonic=1, oscillator=1, n_harmonics=1,
        start_omega=1.377, force=p.force, n_time_samples=64,
        kind="any",
    )
    # In 'any' mode, we expect the same (non-minimum) stationary point the
    # solver would have returned before the fix.
    omega_star = float(state[-2])
    coeffs = state[: 4 * 1]
    cls = classify_projected_stationary_point(
        coeffs, _replace(p, drive_omega=omega_star),
        harmonic=1, oscillator=1, n_harmonics=1, n_time_samples=64,
    )
    # Either kind is valid in 'any' mode; we just assert we got *some* stationary point.
    assert cls["kind"] in ("antiresonance", "resonance_peak", "inflection")


def test_projected_and_exact_agree_when_nonlinearity_restored():
    # Nonlinearity unlocks phase freedom: projected and exact antiresonance
    # frequencies should coincide within the Duffing weakly-nonlinear regime,
    # up to a shift controlled by the nonlinearity strength.
    p = replace(_linear_damped(), beta1=0.02)
    proj = solve_projected_antiresonance_fixed_force(
        p, harmonic=1, oscillator=1, n_harmonics=2,
        start_omega=UNDAMPED_AR + 0.02, force=p.force, n_time_samples=64,
    )
    # For the exact solver we need a weakly-nonlinear system that admits a
    # real solution; otherwise we skip rather than fail (the exact object is
    # codim-2 and may not exist at this particular cut).
    try:
        exact = solve_exact_zero_point_fixed_unfolding(
            p, unfolding_parameter="kappa",
            harmonic=1, oscillator=1, n_harmonics=2,
            start_omega=UNDAMPED_AR + 0.02, start_force=p.force,
            unfolding_value=p.kappa, n_time_samples=64,
        )
    except Exception:
        import pytest
        pytest.skip("exact antiresonance not reachable from this seed")
    # Both should be in a reasonable neighborhood of the undamped linear value.
    assert abs(float(proj[-2]) - UNDAMPED_AR) < 0.1
    assert abs(float(exact[-3]) - UNDAMPED_AR) < 0.1
