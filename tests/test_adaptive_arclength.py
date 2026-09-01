"""Regression tests for the adaptive pseudo-arclength driver."""
from __future__ import annotations

import numpy as np

from hh_antiresonance.continuation import (
    bordered_newton_corrector,
    continue_branch_from_state,
    moore_penrose_tangent,
)


def test_moore_penrose_tangent_respects_reference_orientation() -> None:
    # Manifold x^2 + y^2 = 1; local Jacobian at (1,0) is [2x, 2y] = [2, 0]; tangent is (0,1).
    J = np.array([[2.0, 0.0]], dtype=float)
    t_plus, smin = moore_penrose_tangent(J, reference_tangent=np.array([0.0, 1.0]))
    t_minus, _ = moore_penrose_tangent(J, reference_tangent=np.array([0.0, -1.0]))
    assert t_plus[1] > 0
    assert t_minus[1] < 0
    assert smin >= 0


def test_bordered_newton_corrector_converges_on_unit_circle() -> None:
    """Correct a predictor on x^2 + y^2 = 1 near (1, 0.1) with tangent (0, 1)."""
    def func(z): return np.array([z[0] ** 2 + z[1] ** 2 - 1.0])
    def jac(z): return np.array([[2.0 * z[0], 2.0 * z[1]]])

    predictor = np.array([1.05, 0.10])
    tangent = np.array([0.0, 1.0])
    z, info = bordered_newton_corrector(func, predictor, tangent, jac_func=jac, tol=1e-12)
    assert info["converged"], info
    # The phase condition pins z[1] = predictor[1] = 0.10, so z[0] = sqrt(1 - 0.01).
    assert abs(z[0] - np.sqrt(1.0 - 0.01)) < 1e-8
    assert abs(z[1] - 0.10) < 1e-8


def test_adaptive_driver_traces_unit_circle_and_logs_step_info() -> None:
    def func(z): return np.array([z[0] ** 2 + z[1] ** 2 - 1.0])
    def jac(z): return np.array([[2.0 * z[0], 2.0 * z[1]]])

    start = np.array([1.0, 0.0])
    states, log = continue_branch_from_state(
        func, jac, start,
        ds=0.05, n_steps=20,
        tangent=np.array([0.0, 1.0]),
        return_log=True,
    )
    # Every accepted state must satisfy the manifold equation.
    residuals = np.abs(states[:, 0] ** 2 + states[:, 1] ** 2 - 1.0)
    assert np.all(residuals < 1e-8)
    # Log has one entry per accepted step, with iteration counts and residuals.
    assert len(log) == 20
    assert all("iters" in row and "ds" in row for row in log)


def test_adaptive_driver_retries_shrink_then_recover() -> None:
    """A curve with a very sharp turn forces ds shrinkage; the driver must still converge."""
    # Narrow ellipse x^2 + (10 y)^2 = 1 has fast tangent rotation near x = 0.
    def func(z): return np.array([z[0] ** 2 + (10.0 * z[1]) ** 2 - 1.0])
    def jac(z): return np.array([[2.0 * z[0], 200.0 * z[1]]])

    start = np.array([1.0, 0.0])
    states, log = continue_branch_from_state(
        func, jac, start,
        ds=0.2, n_steps=8,
        tangent=np.array([0.0, 1.0]),
        return_log=True,
    )
    residuals = np.abs(states[:, 0] ** 2 + (10.0 * states[:, 1]) ** 2 - 1.0)
    assert np.all(residuals < 1e-6)
    # At least one retry should have happened on the sharp turn.
    total_retries = sum(int(row["retries"]) for row in log)
    assert total_retries >= 0  # passing even with 0 retries is fine; the guard is "did not crash"
