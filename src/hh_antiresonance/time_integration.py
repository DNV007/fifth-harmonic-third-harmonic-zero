"""Direct time integration of the coupled oscillator and steady-state Fourier extraction.

This module is the canonical closing-the-loop check for any harmonic-balance
claim in the manuscript: at a declared working point, integrate the full ODE
past the transient, FFT-extract the k-th Fourier amplitude of the selected
coordinate, and compare it against the truncated-HB prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

from .models import CoupledOscillatorParams


@dataclass(frozen=True)
class TimeIntegrationReport:
    """Result of a full-ODE steady-state integration at one working point."""
    drive_omega: float
    force: float
    n_transient_periods: int
    n_steady_periods: int
    samples_per_period: int
    # Steady-state slice used for FFT extraction.
    time: np.ndarray
    x1: np.ndarray
    x2: np.ndarray
    # Fourier amplitudes [m=1..max_harmonic] of each coordinate.
    amplitudes_osc1: np.ndarray
    amplitudes_osc2: np.ndarray
    # Residual periodicity: L2 distance between the first and last period of the steady window.
    periodicity_residual: float
    solver_message: str


def _coupled_rhs(t: float, y: np.ndarray, p: CoupledOscillatorParams) -> np.ndarray:
    x1, v1, x2, v2 = y
    drive = p.force * np.cos(p.drive_omega * t)
    dx = x1 - x2
    cubic_coupling = p.kappa_nl * dx ** 3
    dv1 = (
        -2.0 * p.zeta1 * v1
        - p.omega1 ** 2 * x1
        - p.alpha1 * x1 ** 2
        - p.beta1 * x1 ** 3
        - p.kappa * dx
        - cubic_coupling
        + drive
    )
    dv2 = (
        -2.0 * p.zeta2 * v2
        - p.omega2 ** 2 * x2
        - p.alpha2 * x2 ** 2
        - p.beta2 * x2 ** 3
        + p.kappa * dx
        + cubic_coupling
    )
    return np.array([v1, dv1, v2, dv2])


def _extract_fourier_amplitudes(
    signal: np.ndarray,
    *,
    samples_per_period: int,
    max_harmonic: int,
) -> np.ndarray:
    """Real-signal Fourier amplitudes at harmonics 1..max_harmonic.

    The signal is assumed to span an integer number of drive periods and be
    uniformly sampled with `samples_per_period` samples per period.
    """
    n_samples = signal.size
    n_periods = n_samples // samples_per_period
    if n_periods < 1:
        raise ValueError("Need at least one full period of samples for Fourier extraction")
    usable = signal[: n_periods * samples_per_period]
    # Drive frequency normalized so that one period is samples_per_period samples.
    # Hence harmonic m corresponds to frequency m / period = m * (samples_per_period * dt)^{-1}.
    # We compute via explicit correlation with cos and sin to avoid worrying about FFT bin ordering.
    t_index = np.arange(usable.size, dtype=float)
    amplitudes = np.zeros(max_harmonic, dtype=float)
    for m in range(1, max_harmonic + 1):
        angle = 2.0 * np.pi * m * t_index / samples_per_period
        a_m = 2.0 * np.mean(usable * np.cos(angle))
        b_m = 2.0 * np.mean(usable * np.sin(angle))
        amplitudes[m - 1] = float(np.hypot(a_m, b_m))
    return amplitudes


def integrate_steady_state(
    params: CoupledOscillatorParams,
    *,
    max_harmonic: int = 5,
    n_transient_periods: int = 80,
    n_steady_periods: int = 32,
    samples_per_period: int = 256,
    ode_method: Literal["RK45", "DOP853", "Radau", "BDF", "LSODA"] = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    initial_state: np.ndarray | None = None,
) -> TimeIntegrationReport:
    """Integrate the coupled ODE past transient; return steady window + Fourier amplitudes."""
    if params.drive_omega <= 0.0:
        raise ValueError("drive_omega must be positive")
    period = 2.0 * np.pi / params.drive_omega
    n_total_periods = n_transient_periods + n_steady_periods
    t_final = n_total_periods * period

    y0 = np.zeros(4, dtype=float) if initial_state is None else np.asarray(initial_state, dtype=float)

    # Dense sampling grid on the steady window only.
    dt = period / samples_per_period
    steady_start = n_transient_periods * period
    steady_times = steady_start + dt * np.arange(n_steady_periods * samples_per_period, dtype=float)

    sol = solve_ivp(
        _coupled_rhs,
        t_span=(0.0, t_final),
        y0=y0,
        t_eval=steady_times,
        args=(params,),
        method=ode_method,
        rtol=rtol,
        atol=atol,
        dense_output=False,
        max_step=period / 16.0,
    )
    if not sol.success:
        raise RuntimeError(f"Time integration failed: {sol.message}")

    time_slice = sol.t - steady_start  # zero-based time within steady window
    x1 = sol.y[0]
    x2 = sol.y[2]

    amps1 = _extract_fourier_amplitudes(x1, samples_per_period=samples_per_period, max_harmonic=max_harmonic)
    amps2 = _extract_fourier_amplitudes(x2, samples_per_period=samples_per_period, max_harmonic=max_harmonic)

    # Periodicity diagnostic: compare first and last period in the steady window.
    first_period = x1[:samples_per_period]
    last_period = x1[-samples_per_period:]
    periodicity_residual = float(np.linalg.norm(first_period - last_period) / max(np.linalg.norm(first_period), 1e-12))

    return TimeIntegrationReport(
        drive_omega=float(params.drive_omega),
        force=float(params.force),
        n_transient_periods=int(n_transient_periods),
        n_steady_periods=int(n_steady_periods),
        samples_per_period=int(samples_per_period),
        time=time_slice,
        x1=x1,
        x2=x2,
        amplitudes_osc1=amps1,
        amplitudes_osc2=amps2,
        periodicity_residual=periodicity_residual,
        solver_message=sol.message,
    )


def compare_hb_to_time_integration(
    params: CoupledOscillatorParams,
    *,
    harmonic: int = 1,
    oscillator: int = 1,
    n_harmonics: int = 3,
    hb_n_time_samples: int = 256,
    hb_initial_guess: np.ndarray | None = None,
    integrator_kwargs: dict | None = None,
) -> dict[str, float | str]:
    """Run both HB and time integration at one working point; return a summary dict."""
    # Local import avoids a module-level cycle with harmonic_balance <-> models.
    from .harmonic_balance import (
        harmonic_amplitude_squared_from_coeffs,
        solve_harmonic_balance,
    )

    integrator_kwargs = dict(integrator_kwargs or {})
    max_harmonic = max(int(harmonic), 3, integrator_kwargs.pop("max_harmonic", 5))

    try:
        coeffs = solve_harmonic_balance(
            params,
            n_harmonics=n_harmonics,
            initial_guess=hb_initial_guess,
            n_time_samples=hb_n_time_samples,
        )
        hb_amp = float(
            harmonic_amplitude_squared_from_coeffs(
                coeffs, harmonic=harmonic, oscillator=oscillator, n_harmonics=n_harmonics
            )
            ** 0.5
        )
        hb_status = "solved"
    except Exception as exc:
        coeffs = None
        hb_amp = float("nan")
        hb_status = f"hb_failed:{type(exc).__name__}"

    try:
        report = integrate_steady_state(params, max_harmonic=max_harmonic, **integrator_kwargs)
        ode_amps = report.amplitudes_osc1 if oscillator == 1 else report.amplitudes_osc2
        ode_amp = float(ode_amps[harmonic - 1])
        periodicity = float(report.periodicity_residual)
        ode_status = "integrated"
    except Exception as exc:
        ode_amp = float("nan")
        periodicity = float("nan")
        ode_status = f"ode_failed:{type(exc).__name__}"

    if np.isfinite(hb_amp) and np.isfinite(ode_amp):
        denom = max(abs(hb_amp), abs(ode_amp), 1e-14)
        relative_error = float(abs(hb_amp - ode_amp) / denom)
        absolute_error = float(abs(hb_amp - ode_amp))
    else:
        relative_error = float("nan")
        absolute_error = float("nan")

    return {
        "harmonic": int(harmonic),
        "oscillator": int(oscillator),
        "drive_omega": float(params.drive_omega),
        "force": float(params.force),
        "hb_amplitude": hb_amp,
        "ode_amplitude": ode_amp,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "periodicity_residual": periodicity,
        "hb_status": hb_status,
        "ode_status": ode_status,
    }
