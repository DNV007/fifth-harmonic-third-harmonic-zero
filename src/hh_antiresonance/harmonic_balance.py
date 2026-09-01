from __future__ import annotations

from dataclasses import replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from .models import CoupledOscillatorParams

jax.config.update("jax_enable_x64", True)

ParameterName = Literal[
    "omega1",
    "omega2",
    "zeta1",
    "zeta2",
    "alpha1",
    "alpha2",
    "beta1",
    "beta2",
    "kappa",
    "kappa_nl",
    "force",
    "drive_omega",
]
ContinuationParameterName = Literal[
    "omega1",
    "omega2",
    "zeta1",
    "zeta2",
    "alpha1",
    "alpha2",
    "beta1",
    "beta2",
    "kappa",
    "kappa_nl",
]

# Note: 'force' and 'drive_omega' are the last two entries so that several
# call sites can use pvals.at[-2].set(force) and pvals.at[-1].set(drive_omega).
# If you change this order, audit those sites.
PARAMETER_ORDER: tuple[ParameterName, ...] = (
    "omega1",
    "omega2",
    "zeta1",
    "zeta2",
    "alpha1",
    "alpha2",
    "beta1",
    "beta2",
    "kappa",
    "kappa_nl",
    "force",
    "drive_omega",
)
PARAMETER_INDEX = {name: idx for idx, name in enumerate(PARAMETER_ORDER)}
N_DYNAMIC_PARAMETERS = len(PARAMETER_ORDER)


def unpack_coefficients(coeffs: np.ndarray, n_harmonics: int) -> tuple[np.ndarray, np.ndarray]:
    expected = 4 * n_harmonics
    if coeffs.size != expected:
        raise ValueError(f"Expected {expected} coefficients, got {coeffs.size}")
    x1 = coeffs[: 2 * n_harmonics].reshape(n_harmonics, 2)
    x2 = coeffs[2 * n_harmonics :].reshape(n_harmonics, 2)
    return x1, x2


def coefficient_index(*, oscillator: int, harmonic: int, component: str, n_harmonics: int) -> int:
    if oscillator not in (1, 2):
        raise ValueError("oscillator must be 1 or 2")
    if not (1 <= harmonic <= n_harmonics):
        raise ValueError("harmonic out of range")
    if component not in ("cos", "sin"):
        raise ValueError("component must be 'cos' or 'sin'")
    offset = 0 if oscillator == 1 else 2 * n_harmonics
    within = 2 * (harmonic - 1) + (0 if component == "cos" else 1)
    return offset + within


def coefficient_count(n_harmonics: int, *, include_mean: bool = False) -> int:
    """Size of the coefficient vector, with the two mean modes appended last."""
    return 4 * n_harmonics + (2 if include_mean else 0)


def mean_index(*, oscillator: int, n_harmonics: int) -> int:
    """Index of a0^(j).

    The mean modes are appended AFTER the 4*n_harmonics oscillatory
    coefficients, so every existing k>=1 index from `coefficient_index` is
    unchanged whether or not the mean modes are carried.
    """
    if oscillator not in (1, 2):
        raise ValueError("oscillator must be 1 or 2")
    return 4 * n_harmonics + (oscillator - 1)


def target_harmonic_indices(*, oscillator: int, harmonic: int, n_harmonics: int) -> tuple[int, int]:
    return (
        coefficient_index(oscillator=oscillator, harmonic=harmonic, component="cos", n_harmonics=n_harmonics),
        coefficient_index(oscillator=oscillator, harmonic=harmonic, component="sin", n_harmonics=n_harmonics),
    )


def _parameter_array(p: CoupledOscillatorParams) -> np.ndarray:
    return np.array(
        [
            p.omega1,
            p.omega2,
            p.zeta1,
            p.zeta2,
            p.alpha1,
            p.alpha2,
            p.beta1,
            p.beta2,
            p.kappa,
            p.kappa_nl,
            p.force,
            p.drive_omega,
        ],
        dtype=float,
    )


def _replace_param(p: CoupledOscillatorParams, name: ParameterName, value: float) -> CoupledOscillatorParams:
    return replace(p, **{name: float(value)})


def _validate_unfolding_parameter(name: ContinuationParameterName) -> None:
    if name not in PARAMETER_INDEX:
        raise ValueError(f"Unknown parameter '{name}'")
    if name in ("force", "drive_omega"):
        raise ValueError("Use a system parameter, not 'force' or 'drive_omega', for codimension-two continuation")


def _series_state_jax(t: jnp.ndarray, coeffs: jnp.ndarray, omega: float, n_harmonics: int,
                      include_mean: bool = False):
    x1_coeffs = coeffs[: 2 * n_harmonics].reshape((n_harmonics, 2))
    x2_coeffs = coeffs[2 * n_harmonics : 4 * n_harmonics].reshape((n_harmonics, 2))
    x1 = jnp.zeros_like(t)
    v1 = jnp.zeros_like(t)
    a1 = jnp.zeros_like(t)
    x2 = jnp.zeros_like(t)
    v2 = jnp.zeros_like(t)
    a2 = jnp.zeros_like(t)
    for m in range(1, n_harmonics + 1):
        w = m * omega
        c = jnp.cos(w * t)
        s = jnp.sin(w * t)
        aa1, bb1 = x1_coeffs[m - 1]
        aa2, bb2 = x2_coeffs[m - 1]
        x1 = x1 + aa1 * c + bb1 * s
        v1 = v1 - aa1 * w * s + bb1 * w * c
        a1 = a1 - aa1 * w * w * c - bb1 * w * w * s
        x2 = x2 + aa2 * c + bb2 * s
        v2 = v2 - aa2 * w * s + bb2 * w * c
        a2 = a2 - aa2 * w * w * c - bb2 * w * w * s
    if include_mean:
        # k = 0 shifts the displacement only; it carries no velocity or
        # acceleration, which is exactly why it needs its own balance equation.
        x1 = x1 + coeffs[4 * n_harmonics]
        x2 = x2 + coeffs[4 * n_harmonics + 1]
    return x1, v1, a1, x2, v2, a2


def _hb_residual_jax(coeffs: jnp.ndarray, pvals: jnp.ndarray, n_harmonics: int, n_time_samples: int,
                     include_mean: bool = False, force_dc: float = 0.0) -> jnp.ndarray:
    omega1, omega2, zeta1, zeta2, alpha1, alpha2, beta1, beta2, kappa, kappa_nl, force, drive_omega = pvals
    period = 2.0 * jnp.pi / drive_omega
    dt = period / float(n_time_samples)
    t = jnp.arange(n_time_samples, dtype=coeffs.dtype) * dt

    x1, v1, a1, x2, v2, a2 = _series_state_jax(t, coeffs, drive_omega, n_harmonics, include_mean)
    f_drive = force * jnp.cos(drive_omega * t) + force_dc
    dx = x1 - x2
    cubic_coupling = kappa_nl * dx**3
    dyn_a1 = -2.0 * zeta1 * v1 - omega1**2 * x1 - alpha1 * x1**2 - beta1 * x1**3 - kappa * dx - cubic_coupling + f_drive
    dyn_a2 = -2.0 * zeta2 * v2 - omega2**2 * x2 - alpha2 * x2**2 - beta2 * x2**3 + kappa * dx + cubic_coupling
    res1 = a1 - dyn_a1
    res2 = a2 - dyn_a2

    residual = []
    for m in range(1, n_harmonics + 1):
        c = jnp.cos(m * drive_omega * t)
        s = jnp.sin(m * drive_omega * t)
        residual.extend(
            [
                dt * jnp.sum(res1 * c),
                dt * jnp.sum(res1 * s),
                dt * jnp.sum(res2 * c),
                dt * jnp.sum(res2 * s),
            ]
        )
    if include_mean:
        # Zero-frequency balance, appended last so the harmonic row order is
        # unchanged. Without these rows a nonzero alpha_j leaves the mean
        # equation unrepresented and unenforced.
        residual.extend([dt * jnp.sum(res1), dt * jnp.sum(res2)])
    return jnp.asarray(residual)


def harmonic_balance_residual(
    coeffs: np.ndarray,
    p: CoupledOscillatorParams,
    *,
    n_harmonics: int = 3,
    n_time_samples: int = 512,
    include_mean: bool = False,
    force_dc: float = 0.0,
) -> np.ndarray:
    coeffs_arr = jnp.asarray(np.asarray(coeffs, dtype=float))
    pvals = jnp.asarray(_parameter_array(p))
    return np.asarray(
        _hb_residual_jax(coeffs_arr, pvals, n_harmonics, n_time_samples, include_mean, force_dc),
        dtype=float,
    )


def finite_difference_jacobian(func, x: np.ndarray, *, step: float = 1e-7) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(func(x), dtype=float)
    jac = np.zeros((f0.size, x.size), dtype=float)
    for j in range(x.size):
        h = step * max(1.0, abs(x[j]))
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        jac[:, j] = (np.asarray(func(x_plus), dtype=float) - np.asarray(func(x_minus), dtype=float)) / (2.0 * h)
    return jac


def solve_harmonic_balance(
    p: CoupledOscillatorParams,
    *,
    n_harmonics: int = 3,
    initial_guess: np.ndarray | None = None,
    n_time_samples: int = 512,
    tol: float = 1e-10,
    max_nfev: int = 600,
    residual_tol: float | None = None,
    include_mean: bool = False,
    force_dc: float = 0.0,
) -> np.ndarray:
    # `tol` controls LM step convergence (xtol/ftol/gtol on the inner solver),
    # NOT the residual norm: LM can satisfy step-size tolerances at a residual
    # of ~1e-3 and report success. Pass `residual_tol` to additionally enforce
    # ||residual|| <= residual_tol post-solve and raise if violated.
    n_coeffs = coefficient_count(n_harmonics, include_mean=include_mean)
    if initial_guess is None:
        initial_guess = np.zeros(n_coeffs, dtype=float)
        initial_guess[coefficient_index(oscillator=1, harmonic=1, component="cos", n_harmonics=n_harmonics)] = (
            p.force / max(1e-6, abs(p.omega1**2 - p.drive_omega**2))
        )
    initial_guess = np.asarray(initial_guess, dtype=float)
    residual = lambda c: harmonic_balance_residual(
        c, p, n_harmonics=n_harmonics, n_time_samples=n_time_samples,
        include_mean=include_mean, force_dc=force_dc)
    jacobian = lambda c: coefficient_jacobian(
        c, p, n_harmonics=n_harmonics, n_time_samples=n_time_samples,
        include_mean=include_mean, force_dc=force_dc)
    sol = least_squares(residual, initial_guess, jac=jacobian, xtol=tol, ftol=tol, gtol=tol, max_nfev=max_nfev)
    if not sol.success:
        raise RuntimeError(f"Harmonic-balance solve failed: {sol.message}")
    if residual_tol is not None:
        r_norm = float(np.linalg.norm(np.asarray(residual(sol.x))))
        if r_norm > residual_tol:
            raise RuntimeError(
                f"Harmonic-balance solve converged in step-norm but residual "
                f"||r||={r_norm:.3e} exceeds residual_tol={residual_tol:.3e}"
            )
    return np.asarray(sol.x, dtype=float)


def coefficient_jacobian(
    coeffs: np.ndarray,
    p: CoupledOscillatorParams,
    *,
    n_harmonics: int = 3,
    n_time_samples: int = 512,
    include_mean: bool = False,
    force_dc: float = 0.0,
) -> np.ndarray:
    coeffs_arr = jnp.asarray(np.asarray(coeffs, dtype=float))
    pvals = jnp.asarray(_parameter_array(p))
    jac_fun = jax.jacfwd(
        lambda c: _hb_residual_jax(c, pvals, n_harmonics, n_time_samples, include_mean, force_dc))
    return np.asarray(jac_fun(coeffs_arr), dtype=float)


def _hb_state_residual_jax(state: jnp.ndarray, params_template: CoupledOscillatorParams, n_harmonics: int, n_time_samples: int) -> jnp.ndarray:
    n_coeffs = 4 * n_harmonics
    coeffs = state[:n_coeffs]
    omega = state[-2]
    force = state[-1]
    pvals = jnp.asarray(_parameter_array(params_template)).at[-2].set(force).at[-1].set(omega)
    return _hb_residual_jax(coeffs, pvals, n_harmonics, n_time_samples)


def hb_state_jacobian(
    state: np.ndarray,
    params_template: CoupledOscillatorParams,
    *,
    n_harmonics: int = 3,
    n_time_samples: int = 512,
) -> np.ndarray:
    state_arr = jnp.asarray(np.asarray(state, dtype=float))
    jac_fun = jax.jacfwd(lambda z: _hb_state_residual_jax(z, params_template, n_harmonics, n_time_samples))
    return np.asarray(jac_fun(state_arr), dtype=float)


def parameter_partial(
    coeffs: np.ndarray,
    p: CoupledOscillatorParams,
    *,
    param_name: ParameterName,
    n_harmonics: int = 3,
    n_time_samples: int = 512,
    step: float | None = None,
) -> np.ndarray:
    del step
    coeffs_arr = jnp.asarray(np.asarray(coeffs, dtype=float))
    residual_fun = lambda pvals: _hb_residual_jax(coeffs_arr, pvals, n_harmonics, n_time_samples)
    pvals = jnp.asarray(_parameter_array(p))
    pjac = np.asarray(jax.jacfwd(residual_fun)(pvals), dtype=float)
    return pjac[:, PARAMETER_INDEX[param_name]].copy()


def harmonic_amplitude_squared_from_coeffs(
    coeffs: np.ndarray,
    *,
    harmonic: int,
    oscillator: int = 1,
    n_harmonics: int = 3,
) -> float:
    i_cos, i_sin = target_harmonic_indices(oscillator=oscillator, harmonic=harmonic, n_harmonics=n_harmonics)
    return float(coeffs[i_cos] ** 2 + coeffs[i_sin] ** 2)


def amplitude_squared_gradient(
    coeffs: np.ndarray,
    *,
    harmonic: int,
    oscillator: int = 1,
    n_harmonics: int = 3,
) -> np.ndarray:
    grad = np.zeros_like(np.asarray(coeffs, dtype=float))
    i_cos, i_sin = target_harmonic_indices(oscillator=oscillator, harmonic=harmonic, n_harmonics=n_harmonics)
    grad[i_cos] = 2.0 * coeffs[i_cos]
    grad[i_sin] = 2.0 * coeffs[i_sin]
    return grad


def implicit_coefficient_derivative(
    coeffs: np.ndarray,
    p: CoupledOscillatorParams,
    *,
    wrt: ParameterName,
    n_harmonics: int = 3,
    n_time_samples: int = 512,
    step: float | None = None,
) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=float)
    coeff_jac = coefficient_jacobian(coeffs, p, n_harmonics=n_harmonics, n_time_samples=n_time_samples)
    if wrt in ("drive_omega", "force"):
        state = np.concatenate([coeffs, np.array([p.drive_omega, p.force], dtype=float)])
        jac_state = hb_state_jacobian(state, p, n_harmonics=n_harmonics, n_time_samples=n_time_samples)
        rhs_vec = -jac_state[:, -2 if wrt == "drive_omega" else -1]
    else:
        rhs_vec = -parameter_partial(coeffs, p, param_name=wrt, n_harmonics=n_harmonics, n_time_samples=n_time_samples)
    return np.asarray(np.linalg.pinv(coeff_jac) @ rhs_vec, dtype=float)


def _projected_minimum_condition_jax(
    coeffs: jnp.ndarray,
    pvals: jnp.ndarray,
    *,
    harmonic: int = 1,
    oscillator: int = 1,
    n_harmonics: int = 3,
    n_time_samples: int = 512,
) -> jnp.ndarray:
    coeff_jac = jax.jacfwd(lambda c: _hb_residual_jax(c, pvals, n_harmonics, n_time_samples))(coeffs)

    def residual_vs_omega(omega: jnp.ndarray) -> jnp.ndarray:
        pmod = pvals.at[-1].set(omega)
        return _hb_residual_jax(coeffs, pmod, n_harmonics, n_time_samples)

    partial_omega = jax.jacfwd(residual_vs_omega)(pvals[-1])
    dcdomega = -jnp.linalg.pinv(coeff_jac) @ partial_omega
    grad = jnp.zeros_like(coeffs)
    i_cos, i_sin = target_harmonic_indices(oscillator=oscillator, harmonic=harmonic, n_harmonics=n_harmonics)
    grad = grad.at[i_cos].set(2.0 * coeffs[i_cos])
    grad = grad.at[i_sin].set(2.0 * coeffs[i_sin])
    return jnp.dot(grad, dcdomega)


def projected_minimum_condition(
    coeffs: np.ndarray,
    p: CoupledOscillatorParams,
    *,
    harmonic: int = 1,
    oscillator: int = 1,
    n_harmonics: int = 3,
    n_time_samples: int = 512,
    step: float | None = None,
) -> float:
    coeffs_arr = jnp.asarray(np.asarray(coeffs, dtype=float))
    pvals = jnp.asarray(_parameter_array(p))
    return float(
        _projected_minimum_condition_jax(
            coeffs_arr,
            pvals,
            harmonic=harmonic,
            oscillator=oscillator,
            n_harmonics=n_harmonics,
            n_time_samples=n_time_samples,
        )
    )


def classify_projected_stationary_point(
    coeffs: np.ndarray,
    p: CoupledOscillatorParams,
    *,
    harmonic: int = 1,
    oscillator: int = 1,
    n_harmonics: int = 3,
    n_time_samples: int = 512,
    domega: float = 1e-4,
) -> dict[str, float | str]:
    """Classify a projected-antiresonance point as a true minimum, a maximum, or an inflection.

    Computes d^2 A_k^2 / dOmega^2 on the HB manifold by central differencing the
    projected stationarity condition. Returned `second_derivative` is positive at a
    genuine antiresonance (local minimum of the response), negative at a peak
    resonance, and ~0 at an inflection.
    """
    coeffs_arr = jnp.asarray(np.asarray(coeffs, dtype=float))
    pvals = jnp.asarray(_parameter_array(p))
    center = float(pvals[-1])
    plus = pvals.at[-1].set(center + domega)
    minus = pvals.at[-1].set(center - domega)
    g_plus = float(
        _projected_minimum_condition_jax(
            coeffs_arr, plus,
            harmonic=harmonic, oscillator=oscillator,
            n_harmonics=n_harmonics, n_time_samples=n_time_samples,
        )
    )
    g_minus = float(
        _projected_minimum_condition_jax(
            coeffs_arr, minus,
            harmonic=harmonic, oscillator=oscillator,
            n_harmonics=n_harmonics, n_time_samples=n_time_samples,
        )
    )
    second_derivative = (g_plus - g_minus) / (2.0 * domega)
    if second_derivative > 1e-10:
        kind = "antiresonance"
    elif second_derivative < -1e-10:
        kind = "resonance_peak"
    else:
        kind = "inflection"
    return {"second_derivative": second_derivative, "kind": kind}


def _projected_branch_residual_jax(
    state: jnp.ndarray,
    params_template: CoupledOscillatorParams,
    *,
    harmonic: int = 1,
    oscillator: int = 1,
    n_harmonics: int = 3,
    n_time_samples: int = 512,
) -> jnp.ndarray:
    n_coeffs = 4 * n_harmonics
    coeffs = state[:n_coeffs]
    omega = state[-2]
    force = state[-1]
    pvals = jnp.asarray(_parameter_array(params_template)).at[-2].set(force).at[-1].set(omega)
    hb = _hb_residual_jax(coeffs, pvals, n_harmonics, n_time_samples)
    stationarity = _projected_minimum_condition_jax(
        coeffs,
        pvals,
        harmonic=harmonic,
        oscillator=oscillator,
        n_harmonics=n_harmonics,
        n_time_samples=n_time_samples,
    )
    return jnp.concatenate([hb, jnp.array([stationarity], dtype=coeffs.dtype)])


def projected_branch_residual(
    state: np.ndarray,
    params_template: CoupledOscillatorParams,
    *,
    harmonic: int = 1,
    oscillator: int = 1,
    n_harmonics: int = 3,
    n_time_samples: int = 512,
    step: float | None = None,
) -> np.ndarray:
    state_arr = jnp.asarray(np.asarray(state, dtype=float))
    return np.asarray(
        _projected_branch_residual_jax(
            state_arr,
            params_template,
            harmonic=harmonic,
            oscillator=oscillator,
            n_harmonics=n_harmonics,
            n_time_samples=n_time_samples,
        ),
        dtype=float,
    )


def projected_branch_jacobian(
    state: np.ndarray,
    params_template: CoupledOscillatorParams,
    *,
    harmonic: int = 1,
    oscillator: int = 1,
    n_harmonics: int = 3,
    n_time_samples: int = 512,
) -> np.ndarray:
    state_arr = jnp.asarray(np.asarray(state, dtype=float))
    jac_fun = jax.jacfwd(
        lambda z: _projected_branch_residual_jax(
            z,
            params_template,
            harmonic=harmonic,
            oscillator=oscillator,
            n_harmonics=n_harmonics,
            n_time_samples=n_time_samples,
        )
    )
    return np.asarray(jac_fun(state_arr), dtype=float)


def _exact_zero_constraints(coeffs: jnp.ndarray, *, harmonic: int, oscillator: int, n_harmonics: int) -> jnp.ndarray:
    i_cos, i_sin = target_harmonic_indices(oscillator=oscillator, harmonic=harmonic, n_harmonics=n_harmonics)
    return jnp.asarray([coeffs[i_cos], coeffs[i_sin]], dtype=coeffs.dtype)


def _exact_zero_branch_residual_jax(
    state: jnp.ndarray,
    params_template: CoupledOscillatorParams,
    *,
    unfolding_parameter: ContinuationParameterName,
    harmonic: int = 1,
    oscillator: int = 1,
    n_harmonics: int = 3,
    n_time_samples: int = 512,
) -> jnp.ndarray:
    _validate_unfolding_parameter(unfolding_parameter)
    n_coeffs = 4 * n_harmonics
    coeffs = state[:n_coeffs]
    omega = state[-3]
    force = state[-2]
    unfolding_value = state[-1]
    pvals = jnp.asarray(_parameter_array(params_template))
    pvals = pvals.at[PARAMETER_INDEX[unfolding_parameter]].set(unfolding_value)
    pvals = pvals.at[PARAMETER_INDEX["force"]].set(force)
    pvals = pvals.at[PARAMETER_INDEX["drive_omega"]].set(omega)
    hb = _hb_residual_jax(coeffs, pvals, n_harmonics, n_time_samples)
    zero_constraints = _exact_zero_constraints(coeffs, harmonic=harmonic, oscillator=oscillator, n_harmonics=n_harmonics)
    return jnp.concatenate([hb, zero_constraints])


def exact_zero_branch_residual(
    state: np.ndarray,
    params_template: CoupledOscillatorParams,
    *,
    unfolding_parameter: ContinuationParameterName,
    harmonic: int = 1,
    oscillator: int = 1,
    n_harmonics: int = 3,
    n_time_samples: int = 512,
) -> np.ndarray:
    state_arr = jnp.asarray(np.asarray(state, dtype=float))
    return np.asarray(
        _exact_zero_branch_residual_jax(
            state_arr,
            params_template,
            unfolding_parameter=unfolding_parameter,
            harmonic=harmonic,
            oscillator=oscillator,
            n_harmonics=n_harmonics,
            n_time_samples=n_time_samples,
        ),
        dtype=float,
    )


def exact_zero_branch_jacobian(
    state: np.ndarray,
    params_template: CoupledOscillatorParams,
    *,
    unfolding_parameter: ContinuationParameterName,
    harmonic: int = 1,
    oscillator: int = 1,
    n_harmonics: int = 3,
    n_time_samples: int = 512,
) -> np.ndarray:
    state_arr = jnp.asarray(np.asarray(state, dtype=float))
    jac_fun = jax.jacfwd(
        lambda z: _exact_zero_branch_residual_jax(
            z,
            params_template,
            unfolding_parameter=unfolding_parameter,
            harmonic=harmonic,
            oscillator=oscillator,
            n_harmonics=n_harmonics,
            n_time_samples=n_time_samples,
        )
    )
    return np.asarray(jac_fun(state_arr), dtype=float)


def _solve_projected_stationary_once(
    params_template: CoupledOscillatorParams,
    *,
    harmonic: int,
    oscillator: int,
    n_harmonics: int,
    start_omega: float,
    force: float,
    initial_guess: np.ndarray | None,
    n_time_samples: int,
    tol: float,
    max_nfev: int,
) -> np.ndarray:
    """Single attempt: returns (coeffs, omega, force) solving HB + stationarity."""
    p_seed = replace(params_template, drive_omega=float(start_omega), force=float(force))
    coeffs0 = solve_harmonic_balance(
        p_seed,
        n_harmonics=n_harmonics,
        initial_guess=initial_guess,
        n_time_samples=n_time_samples,
        tol=tol,
        max_nfev=max_nfev,
    )
    z0 = np.concatenate([coeffs0, np.array([float(start_omega)], dtype=float)])

    def residual(z: np.ndarray) -> np.ndarray:
        coeffs = z[:-1]
        omega = float(z[-1])
        p = replace(params_template, drive_omega=omega, force=float(force))
        hb = harmonic_balance_residual(coeffs, p, n_harmonics=n_harmonics, n_time_samples=n_time_samples)
        stationarity = projected_minimum_condition(
            coeffs,
            p,
            harmonic=harmonic,
            oscillator=oscillator,
            n_harmonics=n_harmonics,
            n_time_samples=n_time_samples,
        )
        return np.concatenate([hb, np.array([stationarity], dtype=float)])

    def jacobian(z: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(z[:-1], dtype=float)
        omega = float(z[-1])
        state = np.concatenate([coeffs, np.array([omega, force], dtype=float)])
        full_jac = projected_branch_jacobian(
            state,
            params_template,
            harmonic=harmonic,
            oscillator=oscillator,
            n_harmonics=n_harmonics,
            n_time_samples=n_time_samples,
        )
        keep_cols = list(range(4 * n_harmonics)) + [-2]
        return full_jac[:, keep_cols]

    sol = least_squares(residual, z0, jac=jacobian, xtol=tol, ftol=tol, gtol=tol, max_nfev=max_nfev)
    if not sol.success:
        raise RuntimeError(f"Projected antiresonance solve failed: {sol.message}")
    return np.concatenate([np.asarray(sol.x[:-1], dtype=float), np.array([float(sol.x[-1]), float(force)])])


def solve_projected_antiresonance_fixed_force(
    params_template: CoupledOscillatorParams,
    *,
    harmonic: int = 1,
    oscillator: int = 1,
    n_harmonics: int = 3,
    start_omega: float,
    force: float,
    initial_guess: np.ndarray | None = None,
    n_time_samples: int = 512,
    tol: float = 1e-10,
    max_nfev: int = 800,
    kind: Literal["minimum", "any"] = "minimum",
    reseed_fraction: float = 0.04,
) -> np.ndarray:
    """Solve HB + projected-stationarity at fixed force.

    The stationarity condition $\\partial_\\Omega |X_k|^2|_{\\rm HB} = 0$ is satisfied
    by minima, local maxima, and inflections of the response amplitude. By default,
    this solver enforces that the returned point is a local *minimum* of $|X_k|^2$
    (the physical antiresonance). If the initial seed lands on a non-minimum, the
    solver attempts two symmetric reseeds at ``start_omega*(1 +/- reseed_fraction)``
    and raises ``ProjectedAntiresonanceError`` if neither produces a minimum.

    Pass ``kind="any"`` to recover the unchecked original behavior (useful for
    plotting all stationary branches or debugging).
    """
    def _classify(state: np.ndarray) -> str:
        coeffs = state[: 4 * n_harmonics]
        omega_star = float(state[-2])
        p_star = replace(params_template, drive_omega=omega_star, force=float(force))
        info = classify_projected_stationary_point(
            coeffs, p_star,
            harmonic=harmonic, oscillator=oscillator,
            n_harmonics=n_harmonics, n_time_samples=n_time_samples,
        )
        return str(info["kind"])

    state = _solve_projected_stationary_once(
        params_template,
        harmonic=harmonic, oscillator=oscillator, n_harmonics=n_harmonics,
        start_omega=start_omega, force=force,
        initial_guess=initial_guess, n_time_samples=n_time_samples,
        tol=tol, max_nfev=max_nfev,
    )
    if kind == "any":
        return state

    classification = _classify(state)
    if classification == "antiresonance":
        return state

    # Reseed symmetrically to escape the non-minimum basin.
    omega_center = float(state[-2])
    delta = max(abs(omega_center) * reseed_fraction, 1e-3)
    attempts: list[str] = [classification]
    for sign in (-1.0, +1.0):
        try:
            retry = _solve_projected_stationary_once(
                params_template,
                harmonic=harmonic, oscillator=oscillator, n_harmonics=n_harmonics,
                start_omega=omega_center + sign * delta, force=force,
                initial_guess=None, n_time_samples=n_time_samples,
                tol=tol, max_nfev=max_nfev,
            )
        except Exception as exc:
            attempts.append(f"reseed_{sign:+.0f}_failed:{type(exc).__name__}")
            continue
        retry_cls = _classify(retry)
        attempts.append(f"reseed_{sign:+.0f}:{retry_cls}")
        if retry_cls == "antiresonance":
            return retry
    raise ProjectedAntiresonanceError(
        f"Projected solver did not converge to a minimum of |X_{harmonic}|^2. "
        f"Attempts: {attempts}. Seed start_omega={start_omega!r}; "
        f"either seed closer to a known minimum or pass kind='any'."
    )


class ProjectedAntiresonanceError(RuntimeError):
    """Raised when the projected solver cannot locate a minimum stationary point."""


def solve_exact_zero_point_fixed_unfolding(
    params_template: CoupledOscillatorParams,
    *,
    unfolding_parameter: ContinuationParameterName,
    harmonic: int = 1,
    oscillator: int = 1,
    n_harmonics: int = 3,
    start_omega: float,
    start_force: float,
    unfolding_value: float,
    initial_guess: np.ndarray | None = None,
    n_time_samples: int = 512,
    tol: float = 1e-10,
    max_nfev: int = 900,
) -> np.ndarray:
    _validate_unfolding_parameter(unfolding_parameter)
    p_seed = _replace_param(params_template, unfolding_parameter, float(unfolding_value))
    p_seed = replace(p_seed, drive_omega=float(start_omega), force=float(start_force))
    coeffs0 = solve_harmonic_balance(
        p_seed,
        n_harmonics=n_harmonics,
        initial_guess=initial_guess,
        n_time_samples=n_time_samples,
        tol=tol,
        max_nfev=max_nfev,
    )
    z0 = np.concatenate([coeffs0, np.array([float(start_omega), float(start_force)], dtype=float)])

    def residual(z: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(z[:-2], dtype=float)
        omega = float(z[-2])
        force = float(z[-1])
        state = np.concatenate([coeffs, np.array([omega, force, float(unfolding_value)], dtype=float)])
        return exact_zero_branch_residual(
            state,
            params_template,
            unfolding_parameter=unfolding_parameter,
            harmonic=harmonic,
            oscillator=oscillator,
            n_harmonics=n_harmonics,
            n_time_samples=n_time_samples,
        )

    def jacobian(z: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(z[:-2], dtype=float)
        omega = float(z[-2])
        force = float(z[-1])
        state = np.concatenate([coeffs, np.array([omega, force, float(unfolding_value)], dtype=float)])
        full_jac = exact_zero_branch_jacobian(
            state,
            params_template,
            unfolding_parameter=unfolding_parameter,
            harmonic=harmonic,
            oscillator=oscillator,
            n_harmonics=n_harmonics,
            n_time_samples=n_time_samples,
        )
        keep_cols = list(range(4 * n_harmonics)) + [-3, -2]
        return full_jac[:, keep_cols]

    sol = least_squares(residual, z0, jac=jacobian, xtol=tol, ftol=tol, gtol=tol, max_nfev=max_nfev)
    if not sol.success:
        raise RuntimeError(f"Exact zero solve failed: {sol.message}")
    return np.concatenate(
        [
            np.asarray(sol.x[:-2], dtype=float),
            np.array([float(sol.x[-2]), float(sol.x[-1]), float(unfolding_value)], dtype=float),
        ]
    )
