"""Reproducibility driver: regenerate every table and figure.

SUBMISSION is the set of drivers behind the Letter and its Supplemental
Material, in the order of `reproduce.md`, which is the script-to-claim map.

Runs each script in order and fails fast on the first driver that raises.
Invoke from the repository root:

    python scripts/run_all.py

Some drivers take several minutes; the absorber-loss continuation and the
frequency-sweep figure are the slowest.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SUBMISSION = [
    # --- working point, truncation, and the projected-source decomposition ---
    "scripts/run_k3_nondegenerate_exact_zero.py",
    "scripts/run_k3_residual_decomposition.py",
    "scripts/run_k3_analytic_cancellation.py",
    "scripts/run_k3_phase_reduction.py",
    "scripts/run_k3_leading_order_scan.py",
    "scripts/run_k3_modal_resonance.py",
    # --- structural persistence and the physical route ---
    "scripts/run_k3_persistence_renull.py",
    "scripts/run_k3_extended_model_persistence.py",
    "scripts/run_k3_quintic_regularized_root.py",
    "scripts/run_k3_quintic_fold_F.py",
    "scripts/run_k3_mean_mode_corrections.py",
    "scripts/run_k3_gate_path.py",
    "scripts/run_k3_gate_certificates.py",
    "scripts/run_k3_electrostatic_coupling.py",
    "scripts/run_k3_high_q_continuation.py",
    "scripts/run_k3_common_damping_continuation.py",
    # --- fold, weak drive, and the comparison branch ---
    "scripts/run_k3_winding_certificate.py",
    "scripts/run_k3_fold_floquet.py",
    "scripts/run_k3_fold_F_free.py",
    "scripts/run_k3_weak_drive_route.py",
    "scripts/run_k3_weak_drive_energy_bound.py",
    "scripts/run_k3_low_frequency_sheet.py",
    "scripts/run_k3_fifth_harmonic_homotopy.py",
    "scripts/run_k3_homotopy_verification.py",
    "scripts/run_k3_second_model_mechanism.py",
    "scripts/make_k3_existence_figure.py",
    # check_manuscript_consistency.py is intentionally not deposited: it audits
    # the LaTeX sources, which are not part of this deposit.
    "scripts/run_k3_sister_family.py",
    # --- cross-checks ---
    "scripts/run_k3_ode_shooting.py",
    "scripts/run_k3_ode_shooting_drive_pair.py",
    "scripts/run_k3_multimethod_verification.py",
    "scripts/run_k3_two_pathway.py",
    "scripts/run_k3_rlc_second_example.py",
    "scripts/run_k3_suppression.py",
    # --- figures (the mechanism figure reads the continuation data above) ---
    # Both panels of SM Fig. S5: make_k3_branch_vs_F.py reads the weak-drive
    # route and sister-family CSVs written earlier in this list.
    "scripts/make_k3_branch_vs_F.py",
    "scripts/run_k3_notch_vs_zero.py",
    "scripts/run_k3_main_figures.py",
    "scripts/render_k3_mechanism_figure.py",
    "scripts/render_k3_persistence_figure.py",
    "scripts/run_k3_sensitivity.py",
]


def main() -> int:
    for rel in SUBMISSION:
        script = REPO_ROOT / rel
        print(f"\n=== Running {rel} ===", flush=True)
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            print(f"[run_all] Failed at {rel} (exit {result.returncode}).", file=sys.stderr)
            return result.returncode
    print("\n[run_all] All drivers completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
