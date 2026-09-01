from __future__ import annotations

from typing import Callable
from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from .harmonic_balance import (
    ContinuationParameterName,
    exact_zero_branch_jacobian,
    exact_zero_branch_residual,
    finite_difference_jacobian,
    harmonic_amplitude_squared_from_coeffs,
    projected_branch_jacobian,
    projected_branch_residual,
    solve_exact_zero_point_fixed_unfolding,
    solve_projected_antiresonance_fixed_force,
)
from .models import CoupledOscillatorParams

ResidualFunction = Callable[[np.ndarray], np.ndarray]
JacobianFunction = Callable[[np.ndarray], np.ndarray]


def _as_residual_vector(func: ResidualFunction, z: np.ndarray) -> np.ndarray:
    return np.atleast_1d(np.asarray(func(np.asarray(z, dtype=float)), dtype=float))


def _as_jacobian_matrix(func: JacobianFunction | None, z: np.ndarray) -> np.ndarray | None:
    if func is None:
        return None
    return np.asarray(func(np.asarray(z, dtype=float)), dtype=float)


def nullspace_tangent(jacobian: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(np.asarray(jacobian, dtype=float), full_matrices=True)
    tangent = vh[-1]
    norm = np.linalg.norm(tangent)
    if norm < 1e-14:
        raise RuntimeError("Failed to compute continuation tangent")
    return tangent / norm


def orient_tangent(tangent: np.ndarray, reference: np.ndarray | None = None, *, preferred_index: int | None = None) -> np.ndarray:
    tangent = np.asarray(tangent, dtype=float)
    tangent = tangent / np.linalg.norm(tangent)
    if reference is not None and np.linalg.norm(reference) > 0 and np.dot(tangent, reference) < 0:
        tangent = -tangent
    elif preferred_index is not None and tangent[preferred_index] < 0:
        tangent = -tangent
    return tangent


def moore_penrose_tangent(
    jacobian_matrix: np.ndarray,
    reference_tangent: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Null-space tangent via SVD, with sign fixed by continuity.

    Returns (unit tangent, smallest singular value). The smallest singular value
    is a fold-proximity diagnostic: it approaches zero as the branch passes a
    fold, and large persistent values indicate a well-conditioned step.
    """
    u, s, vh = np.linalg.svd(np.asarray(jacobian_matrix, dtype=float), full_matrices=True)
    del u
    tangent = vh[-1]
    norm = np.linalg.norm(tangent)
    if norm < 1e-14:
        raise RuntimeError("SVD failed to produce a null-space tangent")
    tangent = tangent / norm
    if reference_tangent is not None and np.dot(tangent, reference_tangent) < 0.0:
        tangent = -tangent
    return tangent, float(s[-1])


def _bordered_linear_solve(jacobian_matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve J dz = -rhs with a lstsq fallback if J is near-singular."""
    try:
        return np.linalg.solve(jacobian_matrix, rhs)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(jacobian_matrix, rhs, rcond=None)
        return solution


def bordered_newton_corrector(
    func: ResidualFunction,
    predictor: np.ndarray,
    tangent: np.ndarray,
    *,
    jac_func: JacobianFunction | None = None,
    tol: float = 1e-10,
    max_iter: int = 30,
    line_search_iters: int = 25,
    armijo_c: float = 0.25,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Damped-Newton corrector on the bordered system [F(z); t(z - z_pred)] = 0.

    Uses an Armijo backtracking line search on the 2-norm of the augmented
    residual; falls back to a least-squares step when the bordered Jacobian is
    numerically singular (at or near a fold).
    """
    predictor = np.asarray(predictor, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-14:
        raise RuntimeError("Corrector received a zero-length tangent")
    tangent = tangent / tangent_norm

    z = predictor.copy()
    residual_jac_fn = (
        jac_func
        if jac_func is not None
        else lambda zz: finite_difference_jacobian(lambda zzz: _as_residual_vector(func, zzz), zz)
    )

    def augmented_residual(zz: np.ndarray) -> np.ndarray:
        fval = _as_residual_vector(func, zz)
        phase = float(np.dot(zz - predictor, tangent))
        return np.concatenate([fval, np.array([phase], dtype=float)])

    aug_res = augmented_residual(z)
    aug_norm = float(np.linalg.norm(aug_res))

    for it in range(max_iter):
        if aug_norm < tol:
            return z, {"iters": it, "residual": aug_norm, "converged": True, "reason": "tol"}
        J = _as_jacobian_matrix(residual_jac_fn, z)
        if J is None:
            raise RuntimeError("Corrector Jacobian returned None")
        aug_J = np.vstack([J, tangent.reshape(1, -1)])
        try:
            dz = _bordered_linear_solve(aug_J, -aug_res)
        except Exception as exc:
            return z, {"iters": it, "residual": aug_norm, "converged": False, "reason": f"linear_solve:{type(exc).__name__}"}

        alpha = 1.0
        accepted = False
        for _ in range(line_search_iters):
            z_try = z + alpha * dz
            new_res = augmented_residual(z_try)
            new_norm = float(np.linalg.norm(new_res))
            if new_norm <= (1.0 - armijo_c * alpha) * aug_norm + tol:
                z = z_try
                aug_res = new_res
                aug_norm = new_norm
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            return z, {"iters": it + 1, "residual": aug_norm, "converged": False, "reason": "line_search_stalled"}

    return z, {"iters": max_iter, "residual": aug_norm, "converged": aug_norm < tol, "reason": "max_iter"}


def corrector_step(
    func: ResidualFunction,
    predictor: np.ndarray,
    tangent: np.ndarray,
    *,
    jac_func: JacobianFunction | None = None,
    tol: float = 1e-10,
    max_nfev: int = 120,
) -> np.ndarray:
    """Backwards-compatible entry point: damped-Newton corrector with LM fallback.

    `max_nfev` is interpreted as the Newton iteration budget; retained for
    callsite compatibility.
    """
    z, info = bordered_newton_corrector(
        func, predictor, tangent, jac_func=jac_func, tol=tol, max_iter=int(max_nfev),
    )
    if info["converged"]:
        return z

    # LM fallback: treat the bordered system as a square least-squares problem.
    predictor_arr = np.asarray(predictor, dtype=float)
    tangent_arr = np.asarray(tangent, dtype=float)
    tangent_arr = tangent_arr / np.linalg.norm(tangent_arr)

    def residual_vec(zz: np.ndarray) -> np.ndarray:
        manifold = _as_residual_vector(func, zz)
        phase = np.dot(zz - predictor_arr, tangent_arr)
        return np.concatenate([manifold, np.array([phase], dtype=float)])

    def jacobian_mat(zz: np.ndarray) -> np.ndarray:
        J = _as_jacobian_matrix(jac_func, zz)
        if J is None:
            J = finite_difference_jacobian(lambda zzz: _as_residual_vector(func, zzz), zz)
        return np.vstack([J, tangent_arr])

    sol = least_squares(residual_vec, predictor_arr, jac=jacobian_mat, xtol=tol, ftol=tol, gtol=tol, max_nfev=max_nfev)
    if not sol.success:
        raise RuntimeError(f"Corrector failed (Newton + LM): newton={info['reason']}, lm={sol.message}")
    return np.asarray(sol.x, dtype=float)


def estimate_branch_tangents(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    tangents = np.zeros_like(states)
    if len(states) < 2:
        return tangents
    tangents[0] = states[1] - states[0]
    tangents[-1] = states[-1] - states[-2]
    if len(states) > 2:
        tangents[1:-1] = states[2:] - states[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-14] = 1.0
    return tangents / norms


def detect_folds_by_sigma_min(
    sigma_min_series: np.ndarray,
    *,
    threshold: float = 1e-3,
    prominence_ratio: float = 2.0,
) -> np.ndarray:
    """Structural fold detector: local minima of the manifold Jacobian's smallest singular value.

    A fold of the continued manifold is characterized by rank-deficiency of the
    manifold Jacobian, i.e. $\\sigma_{\\min} \\to 0$. This detector flags indices
    where $\\sigma_{\\min}[i]$ is a local minimum, falls below ``threshold``, and
    is smaller than its closest flanking maxima by at least ``prominence_ratio``.

    Complements ``detect_parameter_folds``, which is a turnaround detector on a
    specific coordinate (Omega or F). True folds should register in both
    detectors; disagreement indicates either a spurious sign-change (turnaround
    detector false-positive at a flat region) or a near-fold that does not
    actually pinch the parameter projection (detector-specific blind spot).
    """
    s = np.asarray(sigma_min_series, dtype=float)
    if s.ndim != 1 or s.size < 3:
        return np.array([], dtype=int)
    folds: list[int] = []
    for i in range(1, s.size - 1):
        if not np.isfinite(s[i]):
            continue
        if not (s[i] < s[i - 1] and s[i] < s[i + 1]):
            continue
        if s[i] > threshold:
            continue
        # Prominence: nearest flanking max above threshold*prominence_ratio.
        left_max = max(s[max(0, i - 5): i]) if i > 0 else s[i]
        right_max = max(s[i + 1: min(s.size, i + 6)]) if i < s.size - 1 else s[i]
        flank = max(left_max, right_max)
        if flank < s[i] * prominence_ratio:
            continue
        folds.append(i)
    return np.asarray(folds, dtype=int)


def detect_parameter_folds(parameter_values: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    parameter_values = np.asarray(parameter_values, dtype=float)
    if parameter_values.ndim != 1 or parameter_values.size < 3:
        return np.array([], dtype=int)
    delta = np.diff(parameter_values)
    folds: list[int] = []
    for idx in range(1, parameter_values.size - 1):
        left = delta[idx - 1]
        right = delta[idx]
        if abs(left) < tol or abs(right) < tol:
            continue
        if np.sign(left) != np.sign(right):
            folds.append(idx)
    return np.asarray(folds, dtype=int)


def continue_branch_from_state(
    manifold: ResidualFunction,
    jacobian: JacobianFunction,
    start_state: np.ndarray,
    *,
    ds: float,
    n_steps: int,
    tangent: np.ndarray | None = None,
    preferred_index: int | None = None,
    ds_min: float | None = None,
    ds_max: float | None = None,
    target_iters: int = 4,
    max_retries: int = 6,
    return_log: bool = False,
    newton_tol: float = 1e-10,
    newton_max_iter: int = 30,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, float | int | bool | str]]]:
    """Adaptive pseudo-arclength continuation with Moore-Penrose tangent update.

    Each accepted step uses a damped-Newton corrector on the bordered system.
    On corrector failure the step size ``ds`` is halved and the step is retried
    up to ``max_retries`` times; on easy success (Newton iters < ``target_iters``)
    ``ds`` is grown by 1.5; on hard success (iters > 2*target_iters) ``ds`` is
    shrunk by 0.5. The tangent is refreshed at each accepted step via the
    Moore-Penrose null vector of the true manifold Jacobian, with sign pinned
    to the previous tangent for orientation continuity.

    Returns the sequence of accepted states, plus (optionally) a step log.
    """
    state = np.asarray(start_state, dtype=float)
    ds = float(ds)
    if ds_min is None:
        ds_min = max(1e-6, 1e-3 * abs(ds))
    if ds_max is None:
        ds_max = 10.0 * abs(ds)

    if tangent is None:
        tangent, _ = moore_penrose_tangent(jacobian(state))
        tangent = orient_tangent(tangent, preferred_index=preferred_index)
    else:
        tangent = orient_tangent(tangent)

    states = [state.copy()]
    log: list[dict[str, float | int | bool | str]] = []

    for step_index in range(n_steps):
        retries = 0
        while True:
            predictor = state + ds * tangent
            try:
                corrected, info = bordered_newton_corrector(
                    manifold, predictor, tangent,
                    jac_func=jacobian, tol=newton_tol, max_iter=newton_max_iter,
                )
            except Exception as exc:
                info = {"converged": False, "reason": f"exception:{type(exc).__name__}", "iters": 0, "residual": float("nan")}
                corrected = predictor
            if info["converged"]:
                break
            retries += 1
            ds_prev = ds
            ds = max(ds_min, 0.5 * ds)
            if retries > max_retries or ds_prev <= ds_min:
                raise RuntimeError(
                    f"Adaptive arclength failed at step {step_index}: {info.get('reason', 'unknown')}, "
                    f"last residual {info.get('residual', float('nan'))}, ds={ds_prev}"
                )

        # Refresh tangent from Moore-Penrose null vector at the corrected point.
        try:
            tangent_new, smin = moore_penrose_tangent(jacobian(corrected), reference_tangent=tangent)
        except Exception:
            secant = corrected - state
            tangent_new = secant / (np.linalg.norm(secant) + 1e-14)
            smin = float("nan")

        state = corrected
        tangent = tangent_new
        states.append(state.copy())
        log.append({
            "step": step_index,
            "ds": ds,
            "iters": int(info.get("iters", 0)),
            "residual": float(info.get("residual", float("nan"))),
            "min_singular_value": float(smin),
            "retries": retries,
        })

        # Adaptive ds update for the next predictor.
        iters = int(info.get("iters", 0))
        if iters < target_iters:
            ds = min(ds_max, 1.5 * ds)
        elif iters > 2 * target_iters:
            ds = max(ds_min, 0.5 * ds)

    states_arr = np.asarray(states, dtype=float)
    if return_log:
        return states_arr, log
    return states_arr


def switch_branch_from_fold_state(
    manifold: ResidualFunction,
    jacobian: JacobianFunction,
    fold_state: np.ndarray,
    *,
    ds: float,
    n_steps: int,
    tangent_hint: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    fold_state = np.asarray(fold_state, dtype=float)
    tangent = nullspace_tangent(jacobian(fold_state))
    if tangent_hint is not None:
        tangent = orient_tangent(tangent, reference=tangent_hint)
    plus = continue_branch_from_state(manifold, jacobian, fold_state, ds=ds, n_steps=n_steps, tangent=tangent)
    minus = continue_branch_from_state(manifold, jacobian, fold_state, ds=ds, n_steps=n_steps, tangent=-tangent)
    return {"plus": plus, "minus": minus}


def pseudo_arclength_projected_branch(
    params: CoupledOscillatorParams,
    *,
    harmonic: int = 1,
    oscillator: int = 1,
    n_harmonics: int = 3,
    start_omega: float,
    start_force: float,
    ds: float = 0.01,
    n_steps: int = 20,
    initial_guess: np.ndarray | None = None,
    tangent: np.ndarray | None = None,
    n_time_samples: int = 512,
) -> dict[str, np.ndarray]:
    state = solve_projected_antiresonance_fixed_force(
        params,
        harmonic=harmonic,
        oscillator=oscillator,
        n_harmonics=n_harmonics,
        start_omega=start_omega,
        force=start_force,
        initial_guess=initial_guess,
        n_time_samples=n_time_samples,
    )

    manifold = lambda z: projected_branch_residual(
        z,
        params,
        harmonic=harmonic,
        oscillator=oscillator,
        n_harmonics=n_harmonics,
        n_time_samples=n_time_samples,
    )
    jacobian = lambda z: projected_branch_jacobian(
        z,
        params,
        harmonic=harmonic,
        oscillator=oscillator,
        n_harmonics=n_harmonics,
        n_time_samples=n_time_samples,
    )

    states_arr, log = continue_branch_from_state(
        manifold,
        jacobian,
        state,
        ds=ds,
        n_steps=n_steps,
        tangent=tangent,
        preferred_index=-1,
        return_log=True,
    )
    amplitudes = np.asarray(
        [
            harmonic_amplitude_squared_from_coeffs(row[:-2], harmonic=harmonic, oscillator=oscillator, n_harmonics=n_harmonics) ** 0.5
            for row in states_arr
        ],
        dtype=float,
    )
    sigma_min_series = np.asarray(
        [entry.get("min_singular_value", float("nan")) for entry in log],
        dtype=float,
    )
    # log has n_steps entries (one per accepted step); shifted by 1 to match states_arr[1:].
    omega_folds_turn = detect_parameter_folds(states_arr[:, -2])
    force_folds_turn = detect_parameter_folds(states_arr[:, -1])
    structural_folds = detect_folds_by_sigma_min(sigma_min_series) + 1  # offset into states_arr
    return {
        "state": states_arr,
        "omega": states_arr[:, -2].copy(),
        "force": states_arr[:, -1].copy(),
        "amplitude": amplitudes,
        "tangent": estimate_branch_tangents(states_arr),
        "omega_fold_index": omega_folds_turn,
        "force_fold_index": force_folds_turn,
        "structural_fold_index": structural_folds,
        "sigma_min": sigma_min_series,
        "log": log,
    }


def pseudo_arclength_exact_zero_codim_two_branch(
    params: CoupledOscillatorParams,
    *,
    unfolding_parameter: ContinuationParameterName,
    harmonic: int = 1,
    oscillator: int = 1,
    n_harmonics: int = 3,
    start_omega: float,
    start_force: float,
    start_unfolding: float,
    ds: float = 0.01,
    n_steps: int = 20,
    initial_guess: np.ndarray | None = None,
    tangent: np.ndarray | None = None,
    n_time_samples: int = 512,
) -> dict[str, np.ndarray]:
    state = solve_exact_zero_point_fixed_unfolding(
        params,
        unfolding_parameter=unfolding_parameter,
        harmonic=harmonic,
        oscillator=oscillator,
        n_harmonics=n_harmonics,
        start_omega=start_omega,
        start_force=start_force,
        unfolding_value=start_unfolding,
        initial_guess=initial_guess,
        n_time_samples=n_time_samples,
    )

    manifold = lambda z: exact_zero_branch_residual(
        z,
        params,
        unfolding_parameter=unfolding_parameter,
        harmonic=harmonic,
        oscillator=oscillator,
        n_harmonics=n_harmonics,
        n_time_samples=n_time_samples,
    )
    jacobian = lambda z: exact_zero_branch_jacobian(
        z,
        params,
        unfolding_parameter=unfolding_parameter,
        harmonic=harmonic,
        oscillator=oscillator,
        n_harmonics=n_harmonics,
        n_time_samples=n_time_samples,
    )

    states_arr, log = continue_branch_from_state(
        manifold,
        jacobian,
        state,
        ds=ds,
        n_steps=n_steps,
        tangent=tangent,
        preferred_index=-1,
        return_log=True,
    )
    amplitudes = np.asarray(
        [
            harmonic_amplitude_squared_from_coeffs(row[:-3], harmonic=harmonic, oscillator=oscillator, n_harmonics=n_harmonics) ** 0.5
            for row in states_arr
        ],
        dtype=float,
    )
    sigma_min_series = np.asarray(
        [entry.get("min_singular_value", float("nan")) for entry in log],
        dtype=float,
    )
    structural_folds = detect_folds_by_sigma_min(sigma_min_series) + 1
    return {
        "state": states_arr,
        "omega": states_arr[:, -3].copy(),
        "force": states_arr[:, -2].copy(),
        unfolding_parameter: states_arr[:, -1].copy(),
        "amplitude": amplitudes,
        "tangent": estimate_branch_tangents(states_arr),
        "omega_fold_index": detect_parameter_folds(states_arr[:, -3]),
        "force_fold_index": detect_parameter_folds(states_arr[:, -2]),
        f"{unfolding_parameter}_fold_index": detect_parameter_folds(states_arr[:, -1]),
        "structural_fold_index": structural_folds,
        "sigma_min": sigma_min_series,
        "log": log,
    }


def switch_projected_branch_at_detected_fold(
    branch: dict[str, np.ndarray],
    params: CoupledOscillatorParams,
    *,
    fold_index: int,
    harmonic: int = 1,
    oscillator: int = 1,
    n_harmonics: int = 3,
    ds: float = 0.01,
    n_steps: int = 12,
    n_time_samples: int = 512,
) -> dict[str, np.ndarray]:
    states = np.asarray(branch["state"], dtype=float)
    fold_state = states[fold_index]
    tangent_hint = None
    if "tangent" in branch and len(branch["tangent"]) > fold_index:
        tangent_hint = np.asarray(branch["tangent"][fold_index], dtype=float)
    manifold = lambda z: projected_branch_residual(
        z,
        params,
        harmonic=harmonic,
        oscillator=oscillator,
        n_harmonics=n_harmonics,
        n_time_samples=n_time_samples,
    )
    jacobian = lambda z: projected_branch_jacobian(
        z,
        params,
        harmonic=harmonic,
        oscillator=oscillator,
        n_harmonics=n_harmonics,
        n_time_samples=n_time_samples,
    )
    switched = switch_branch_from_fold_state(manifold, jacobian, fold_state, ds=ds, n_steps=n_steps, tangent_hint=tangent_hint)
    out: dict[str, np.ndarray] = {}
    for key, seg in switched.items():
        out[key] = seg
        out[f"{key}_omega"] = seg[:, -2].copy()
        out[f"{key}_force"] = seg[:, -1].copy()
        out[f"{key}_amplitude"] = np.asarray(
            [harmonic_amplitude_squared_from_coeffs(row[:-2], harmonic=harmonic, oscillator=oscillator, n_harmonics=n_harmonics) ** 0.5 for row in seg],
            dtype=float,
        )
    return out


def switch_exact_zero_branch_at_detected_fold(
    branch: dict[str, np.ndarray],
    params: CoupledOscillatorParams,
    *,
    unfolding_parameter: ContinuationParameterName,
    fold_index: int,
    harmonic: int = 1,
    oscillator: int = 1,
    n_harmonics: int = 3,
    ds: float = 0.01,
    n_steps: int = 12,
    n_time_samples: int = 512,
) -> dict[str, np.ndarray]:
    """Switch branch directly on the codimension-two exact-zero manifold.

    The state ordering is [coeffs..., omega, force, unfolding].
    """
    states = np.asarray(branch["state"], dtype=float)
    fold_state = states[fold_index]
    tangent_hint = None
    if "tangent" in branch and len(branch["tangent"]) > fold_index:
        tangent_hint = np.asarray(branch["tangent"][fold_index], dtype=float)

    manifold = lambda z: exact_zero_branch_residual(
        z,
        params,
        unfolding_parameter=unfolding_parameter,
        harmonic=harmonic,
        oscillator=oscillator,
        n_harmonics=n_harmonics,
        n_time_samples=n_time_samples,
    )
    jacobian = lambda z: exact_zero_branch_jacobian(
        z,
        params,
        unfolding_parameter=unfolding_parameter,
        harmonic=harmonic,
        oscillator=oscillator,
        n_harmonics=n_harmonics,
        n_time_samples=n_time_samples,
    )
    switched = switch_branch_from_fold_state(manifold, jacobian, fold_state, ds=ds, n_steps=n_steps, tangent_hint=tangent_hint)
    out: dict[str, np.ndarray] = {}
    for key, seg in switched.items():
        out[key] = seg
        out[f"{key}_omega"] = seg[:, -3].copy()
        out[f"{key}_force"] = seg[:, -2].copy()
        out[f"{key}_{unfolding_parameter}"] = seg[:, -1].copy()
        out[f"{key}_amplitude"] = np.asarray(
            [harmonic_amplitude_squared_from_coeffs(row[:-3], harmonic=harmonic, oscillator=oscillator, n_harmonics=n_harmonics) ** 0.5 for row in seg],
            dtype=float,
        )
    return out


def _effective_detuning_reduced(params: CoupledOscillatorParams, omega: float, kappa: float) -> float:
    denom = (params.omega2**2 + kappa - omega**2)
    if abs(denom) < 1e-8:
        denom = 1e-8 if denom >= 0 else -1e-8
    omega_eff_sq = params.omega1**2 + kappa - (kappa**2) / denom
    return omega_eff_sq - omega**2


def hunt_reduced_duffing_saddle_nodes(
    params: CoupledOscillatorParams,
    *,
    force_values: np.ndarray | list[float],
    beta1_values: np.ndarray | list[float],
    kappa_values: np.ndarray | list[float],
    start_omega: float | None = None,
    omega_window: tuple[float, float] | None = None,
    n_omega_samples: int = 121,
    force_tolerance: float = 0.015,
    **_ignored,
) -> list[dict[str, float | int | np.ndarray]]:
    """Coarse saddle-node seeds over (F, beta1, kappa) from the reduced-order Duffing model.

    Uses adiabatic elimination of oscillator 2 to collapse the coupled system to a
    single-DOF Duffing oscillator with an effective detuning Delta(Omega, kappa).
    Saddle-nodes of the resulting frequency-response curve are returned as
    candidate seeds for the full projected-HB continuation; this is a
    preconditioner on the *reduced* model, not a fold claim on the full system.
    All continuation control parameters (ds, n_steps, n_time_samples, detect_on,
    harmonic, oscillator, n_harmonics) are accepted and ignored here for
    call-site compatibility with the full continuation API.
    """
    force_values = np.asarray(force_values, dtype=float)
    beta1_values = np.asarray(beta1_values, dtype=float)
    kappa_values = np.asarray(kappa_values, dtype=float)
    omega_center = float(params.omega1 if start_omega is None else start_omega)
    if omega_window is None:
        omega_window = (0.8 * omega_center, 1.2 * omega_center)
    omegas = np.linspace(float(omega_window[0]), float(omega_window[1]), int(n_omega_samples))

    candidates: list[dict[str, float | int | np.ndarray]] = []
    for beta1 in beta1_values:
        c = 0.75 * float(beta1)
        if abs(c) < 1e-12:
            continue
        for kappa in kappa_values:
            for omega in omegas:
                Delta = _effective_detuning_reduced(params, float(omega), float(kappa))
                Gamma = 2.0 * params.zeta1 * float(omega)
                # Fold condition for the reduced response curve: 3 c^2 y^2 + 4 c Delta y + Delta^2 + Gamma^2 = 0
                coeff = np.array([3.0 * c * c, 4.0 * c * Delta, Delta * Delta + Gamma * Gamma], dtype=float)
                roots = np.roots(coeff)
                for root in roots:
                    if abs(root.imag) > 1e-10:
                        continue
                    y = float(root.real)
                    if y <= 0.0:
                        continue
                    amp = y ** 0.5
                    response_sq = y * ((Delta + c * y) ** 2 + Gamma * Gamma)
                    if response_sq <= 0.0:
                        continue
                    F_fold = response_sq ** 0.5
                    nearest_force = float(force_values[np.argmin(np.abs(force_values - F_fold))])
                    mismatch = abs(nearest_force - F_fold)
                    if mismatch > force_tolerance:
                        continue
                    # Score combines proximity to requested F with nonlinear prominence.
                    score = float((amp ** 2) / (mismatch + 1e-6))
                    candidates.append({
                        "initial_force": nearest_force,
                        "beta1": float(beta1),
                        "kappa": float(kappa),
                        "fold_index": 0,
                        "omega_at_fold": float(omega),
                        "force_at_fold": float(F_fold),
                        "amplitude_at_fold": float(amp),
                        "score": score,
                    })
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    return candidates




def _full_projected_branch_candidate_task(args: dict[str, object]) -> dict[str, object]:
    params = args["params"]
    branch = pseudo_arclength_projected_branch(
        params,
        harmonic=int(args["harmonic"]),
        oscillator=int(args["oscillator"]),
        n_harmonics=int(args["full_n_harmonics"]),
        start_omega=float(args["candidate_omega"]),
        start_force=float(args["candidate_force"]),
        ds=float(args["full_ds"]),
        n_steps=int(args["full_n_steps"]),
        n_time_samples=int(args["full_n_time_samples"]),
    )
    return {
        "omega": np.asarray(branch["omega"], dtype=float),
        "force": np.asarray(branch["force"], dtype=float),
        "omega_fold_index": np.asarray(branch["omega_fold_index"], dtype=int),
        "force_fold_index": np.asarray(branch["force_fold_index"], dtype=int),
    }


def validate_reduced_fold_candidates_with_full_projected_branches(
    params: CoupledOscillatorParams,
    *,
    force_values: np.ndarray | list[float],
    beta1_values: np.ndarray | list[float],
    kappa_values: np.ndarray | list[float],
    harmonic: int = 1,
    oscillator: int = 1,
    reduced_n_harmonics: int = 1,
    full_n_harmonics: int = 3,
    start_omega: float | None = None,
    top_n: int = 10,
    reduced_n_omega_samples: int = 121,
    reduced_force_tolerance: float = 0.015,
    full_ds: float = 0.012,
    full_n_steps: int = 12,
    full_n_time_samples: int = 128,
    deduplicate: bool = True,
    candidate_timeout_seconds: float | None = None,
) -> list[dict[str, float | int | bool | str]]:
    """Validate top reduced-model fold candidates with the full projected HB branch solver."""
    reduced_candidates = hunt_reduced_duffing_saddle_nodes(
        params,
        force_values=force_values,
        beta1_values=beta1_values,
        kappa_values=kappa_values,
        harmonic=harmonic,
        oscillator=oscillator,
        n_harmonics=reduced_n_harmonics,
        start_omega=start_omega,
        n_omega_samples=reduced_n_omega_samples,
        force_tolerance=reduced_force_tolerance,
    )
    if deduplicate:
        seen: set[tuple[float, float, float]] = set()
        unique = []
        for cand in reduced_candidates:
            key = (round(float(cand["initial_force"]), 6), round(float(cand["beta1"]), 6), round(float(cand["kappa"]), 6))
            if key in seen:
                continue
            seen.add(key)
            unique.append(cand)
        reduced_candidates = unique

    results = []
    for reduced_rank, cand in enumerate(reduced_candidates[:max(0, int(top_n))], start=1):
        cand_force = float(cand["initial_force"])
        cand_beta1 = float(cand["beta1"])
        cand_kappa = float(cand["kappa"])
        cand_omega = float(cand["omega_at_fold"])
        reduced_score = float(cand["score"])
        record = {
            "reduced_rank": reduced_rank,
            "reduced_score": reduced_score,
            "beta1": cand_beta1,
            "kappa": cand_kappa,
            "initial_force": cand_force,
            "candidate_omega": cand_omega,
            "candidate_force_fold": float(cand["force_at_fold"]),
            "candidate_amplitude": float(cand["amplitude_at_fold"]),
            "full_success": False,
            "survived_full_continuation": False,
            "verified_fold": False,
            "branch_points": 0,
            "n_force_folds": 0,
            "n_omega_folds": 0,
            "closest_force_fold": float("nan"),
            "closest_omega_fold": float("nan"),
            "force_fold_mismatch": float("inf"),
            "omega_fold_mismatch": float("inf"),
            "validation_score": 0.0,
            "status": "not_run",
        }
        full_params = replace(params, beta1=cand_beta1, kappa=cand_kappa, force=cand_force, drive_omega=cand_omega)
        try:
            task_args = {
                "params": full_params,
                "harmonic": harmonic,
                "oscillator": oscillator,
                "full_n_harmonics": full_n_harmonics,
                "candidate_omega": cand_omega,
                "candidate_force": cand_force,
                "full_ds": full_ds,
                "full_n_steps": full_n_steps,
                "full_n_time_samples": full_n_time_samples,
            }
            branch = _full_projected_branch_candidate_task(task_args)
            omega_vals = np.asarray(branch["omega"], dtype=float)
            force_vals = np.asarray(branch["force"], dtype=float)
            omega_folds = np.asarray(branch["omega_fold_index"], dtype=int)
            force_folds = np.asarray(branch["force_fold_index"], dtype=int)
            n_force_folds = int(force_folds.size)
            n_omega_folds = int(omega_folds.size)
            closest_force_fold = float("nan")
            force_mismatch = float("inf")
            if n_force_folds:
                fold_vals = force_vals[force_folds]
                idx = int(np.argmin(np.abs(fold_vals - float(cand["force_at_fold"]))))
                closest_force_fold = float(fold_vals[idx])
                force_mismatch = abs(closest_force_fold - float(cand["force_at_fold"]))
            closest_omega_fold = float("nan")
            omega_mismatch = float("inf")
            if n_omega_folds:
                fold_vals = omega_vals[omega_folds]
                idx = int(np.argmin(np.abs(fold_vals - cand_omega)))
                closest_omega_fold = float(fold_vals[idx])
                omega_mismatch = abs(closest_omega_fold - cand_omega)
            verified = bool(n_force_folds + n_omega_folds)
            length_bonus = 1.0 + 0.02 * max(0, len(omega_vals) - 1)
            fold_bonus = 1.0 + 0.75 * n_force_folds + 0.35 * n_omega_folds
            mismatch_penalty = 1.0 + min(force_mismatch, 1e6) + min(omega_mismatch, 1e6)
            validation_score = (reduced_score * length_bonus * fold_bonus / mismatch_penalty) if verified else (0.2 * reduced_score * length_bonus)
            record.update({
                "full_success": True,
                "survived_full_continuation": True,
                "verified_fold": verified,
                "branch_points": int(len(omega_vals)),
                "n_force_folds": n_force_folds,
                "n_omega_folds": n_omega_folds,
                "closest_force_fold": closest_force_fold,
                "closest_omega_fold": closest_omega_fold,
                "force_fold_mismatch": force_mismatch,
                "omega_fold_mismatch": omega_mismatch,
                "validation_score": float(validation_score),
                "status": "verified_fold" if verified else "continued_no_fold",
            })
        except Exception as exc:
            record.update({
                "status": f"failed:{type(exc).__name__}",
                "error_message": str(exc),
            })
        results.append(record)
    results.sort(key=lambda r: (int(bool(r.get("verified_fold", False))), int(bool(r.get("survived_full_continuation", False))), float(r.get("validation_score", 0.0)), float(r.get("reduced_score", 0.0))), reverse=True)
    for validated_rank, rec in enumerate(results, start=1):
        rec["validated_rank"] = validated_rank
    return results



def adaptive_validate_reduced_hits_with_escalation(
    params: CoupledOscillatorParams,
    *,
    force_values: np.ndarray | list[float],
    beta1_values: np.ndarray | list[float],
    kappa_values: np.ndarray | list[float],
    harmonic: int = 1,
    oscillator: int = 1,
    reduced_n_harmonics: int = 1,
    start_omega: float | None = None,
    top_n: int = 10,
    reduced_n_omega_samples: int = 121,
    reduced_force_tolerance: float = 0.015,
    deduplicate: bool = True,
    escalation_stages: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Run an adaptive reduced-to-full validation campaign.

    The campaign first rank-orders reduced-model fold hits, then promotes only
    those candidates that survive the current projected-HB stage to the next,
    more expensive stage. The intended use is a cheap-to-expensive sequence in
    ``n_harmonics``, ``n_steps``, and ``n_time_samples``.
    """
    if escalation_stages is None:
        escalation_stages = [
            {"label": "cheap", "n_harmonics": 1, "n_steps": 2, "n_time_samples": 16, "ds": 0.005},
            {"label": "medium", "n_harmonics": 2, "n_steps": 4, "n_time_samples": 32, "ds": 0.008},
            {"label": "full", "n_harmonics": 3, "n_steps": 8, "n_time_samples": 64, "ds": 0.012},
        ]
    if len(escalation_stages) == 0:
        raise ValueError("escalation_stages must contain at least one stage")

    reduced_candidates = hunt_reduced_duffing_saddle_nodes(
        params,
        force_values=force_values,
        beta1_values=beta1_values,
        kappa_values=kappa_values,
        harmonic=harmonic,
        oscillator=oscillator,
        n_harmonics=reduced_n_harmonics,
        start_omega=start_omega,
        n_omega_samples=reduced_n_omega_samples,
        force_tolerance=reduced_force_tolerance,
    )
    if deduplicate:
        seen: set[tuple[float, float, float]] = set()
        unique: list[dict[str, object]] = []
        for cand in reduced_candidates:
            key = (round(float(cand["initial_force"]), 6), round(float(cand["beta1"]), 6), round(float(cand["kappa"]), 6))
            if key in seen:
                continue
            seen.add(key)
            unique.append(cand)
        reduced_candidates = unique

    results: list[dict[str, object]] = []
    for reduced_rank, cand in enumerate(reduced_candidates[:max(0, int(top_n))], start=1):
        cand_force = float(cand["initial_force"])
        cand_beta1 = float(cand["beta1"])
        cand_kappa = float(cand["kappa"])
        cand_omega = float(cand["omega_at_fold"])
        reduced_score = float(cand["score"])
        full_params = replace(params, beta1=cand_beta1, kappa=cand_kappa, force=cand_force, drive_omega=cand_omega)

        stage_results: list[dict[str, object]] = []
        status_parts: list[str] = []
        stage_failed = False
        for stage_index, stage in enumerate(escalation_stages, start=1):
            label = str(stage.get("label", f"stage_{stage_index}"))
            try:
                branch = pseudo_arclength_projected_branch(
                    full_params,
                    harmonic=harmonic,
                    oscillator=oscillator,
                    n_harmonics=int(stage.get("n_harmonics", 1)),
                    start_omega=cand_omega,
                    start_force=cand_force,
                    ds=float(stage.get("ds", 0.01)),
                    n_steps=int(stage.get("n_steps", 4)),
                    n_time_samples=int(stage.get("n_time_samples", 32)),
                )
                omega_vals = np.asarray(branch["omega"], dtype=float)
                force_vals = np.asarray(branch["force"], dtype=float)
                omega_folds = np.asarray(branch["omega_fold_index"], dtype=int)
                force_folds = np.asarray(branch["force_fold_index"], dtype=int)
                n_force_folds = int(force_folds.size)
                n_omega_folds = int(omega_folds.size)
                closest_force_fold = float("nan")
                force_mismatch = float("inf")
                if n_force_folds:
                    fold_vals = force_vals[force_folds]
                    idx = int(np.argmin(np.abs(fold_vals - float(cand["force_at_fold"]))))
                    closest_force_fold = float(fold_vals[idx])
                    force_mismatch = abs(closest_force_fold - float(cand["force_at_fold"]))
                closest_omega_fold = float("nan")
                omega_mismatch = float("inf")
                if n_omega_folds:
                    fold_vals = omega_vals[omega_folds]
                    idx = int(np.argmin(np.abs(fold_vals - cand_omega)))
                    closest_omega_fold = float(fold_vals[idx])
                    omega_mismatch = abs(closest_omega_fold - cand_omega)
                verified = bool(n_force_folds + n_omega_folds)
                length_bonus = 1.0 + 0.02 * max(0, len(omega_vals) - 1)
                fold_bonus = 1.0 + 0.75 * n_force_folds + 0.35 * n_omega_folds
                mismatch_penalty = 1.0 + min(force_mismatch, 1e6) + min(omega_mismatch, 1e6)
                validation_score = (reduced_score * length_bonus * fold_bonus / mismatch_penalty) if verified else (0.2 * reduced_score * length_bonus)
                stage_result = {
                    "label": label,
                    "stage_index": stage_index,
                    "status": "verified_fold" if verified else "continued_no_fold",
                    "survived": True,
                    "verified_fold": verified,
                    "branch_points": int(len(omega_vals)),
                    "n_force_folds": n_force_folds,
                    "n_omega_folds": n_omega_folds,
                    "closest_force_fold": closest_force_fold,
                    "closest_omega_fold": closest_omega_fold,
                    "force_fold_mismatch": force_mismatch,
                    "omega_fold_mismatch": omega_mismatch,
                    "validation_score": float(validation_score),
                    "n_harmonics": int(stage.get("n_harmonics", 1)),
                    "n_steps": int(stage.get("n_steps", 4)),
                    "n_time_samples": int(stage.get("n_time_samples", 32)),
                }
                stage_results.append(stage_result)
                status_parts.append(f"{label}:{stage_result['status']}")
            except Exception as exc:
                stage_results.append({
                    "label": label,
                    "stage_index": stage_index,
                    "status": f"failed:{type(exc).__name__}",
                    "survived": False,
                    "verified_fold": False,
                    "branch_points": 0,
                    "n_force_folds": 0,
                    "n_omega_folds": 0,
                    "closest_force_fold": float("nan"),
                    "closest_omega_fold": float("nan"),
                    "force_fold_mismatch": float("inf"),
                    "omega_fold_mismatch": float("inf"),
                    "validation_score": 0.0,
                    "n_harmonics": int(stage.get("n_harmonics", 1)),
                    "n_steps": int(stage.get("n_steps", 4)),
                    "n_time_samples": int(stage.get("n_time_samples", 32)),
                    "error_message": str(exc),
                })
                status_parts.append(f"{label}:failed")
                stage_failed = True
                break
        completed = [s for s in stage_results if bool(s.get("survived", False))]
        deepest = completed[-1] if completed else stage_results[0]
        reached_final_stage = bool(completed) and int(deepest.get("stage_index", 0)) == len(escalation_stages)
        record: dict[str, object] = {
            "reduced_rank": reduced_rank,
            "reduced_score": reduced_score,
            "beta1": cand_beta1,
            "kappa": cand_kappa,
            "initial_force": cand_force,
            "candidate_omega": cand_omega,
            "candidate_force_fold": float(cand["force_at_fold"]),
            "candidate_amplitude": float(cand["amplitude_at_fold"]),
            "stages_attempted": len(stage_results),
            "stages_completed": len(completed),
            "deepest_stage_label": deepest.get("label", ""),
            "deepest_stage_index": int(deepest.get("stage_index", 0)),
            "survived_full_continuation": bool(completed),
            "reached_final_stage": reached_final_stage,
            "verified_fold": bool(deepest.get("verified_fold", False)),
            "branch_points": int(deepest.get("branch_points", 0)),
            "n_force_folds": int(deepest.get("n_force_folds", 0)),
            "n_omega_folds": int(deepest.get("n_omega_folds", 0)),
            "closest_force_fold": float(deepest.get("closest_force_fold", float("nan"))),
            "closest_omega_fold": float(deepest.get("closest_omega_fold", float("nan"))),
            "force_fold_mismatch": float(deepest.get("force_fold_mismatch", float("inf"))),
            "omega_fold_mismatch": float(deepest.get("omega_fold_mismatch", float("inf"))),
            "validation_score": float(deepest.get("validation_score", 0.0)),
            "status": str(deepest.get("status", "not_run")),
            "stage_trace": ";".join(status_parts),
            "error_message": str(stage_results[-1].get("error_message", "")) if stage_failed else "",
            "stage_results": stage_results,
        }
        results.append(record)

    results.sort(
        key=lambda r: (
            int(bool(r.get("reached_final_stage", False))),
            int(bool(r.get("verified_fold", False))),
            int(bool(r.get("survived_full_continuation", False))),
            int(r.get("stages_completed", 0)),
            float(r.get("validation_score", 0.0)),
            float(r.get("reduced_score", 0.0)),
        ),
        reverse=True,
    )
    for adaptive_rank, rec in enumerate(results, start=1):
        rec["adaptive_rank"] = adaptive_rank
    return results
