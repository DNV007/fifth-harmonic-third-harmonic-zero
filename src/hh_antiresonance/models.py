from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoupledOscillatorParams:
    omega1: float
    omega2: float
    zeta1: float
    zeta2: float
    alpha1: float
    alpha2: float
    beta1: float
    beta2: float
    kappa: float
    force: float
    drive_omega: float = 1.0
    kappa_nl: float = 0.0
