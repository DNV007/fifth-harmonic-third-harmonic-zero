# Reproducibility map

Manuscript: *Fifth-Harmonic Feedback Controls the Existence of a
Third-Harmonic Zero*

Every quantity reported in the Letter and its Supplemental Material has a driver
listed below, and the printed values are the ones these drivers produce.

Section, figure, table and equation numbers below are those of the submitted
Letter and its Supplemental Material. Letter items are unprefixed; Supplemental
items carry the `S` prefix used in that document.

## Environment

Python 3.11.5, Linux 5.15 (x86_64), double precision (`JAX_ENABLE_X64=1`).
NumPy 2.4.6, SciPy 1.17.1, JAX/jaxlib 0.10.2, Matplotlib 3.11.0, pandas 3.0.3.

These exact versions are pinned in `pyproject.toml` and resolved in `uv.lock`;
recreate the environment with `uv sync`. Run drivers from the root of this
deposit as

```bash
JAX_ENABLE_X64=1 uv run python scripts/<driver>.py
```

Several drivers report to standard output rather than to a file; those rows say
`stdout`. Redirect them to a file if you want to diff against the paper.

Figure drivers read only from `data/` and write only to `figures/`. The copies
in `figures/` are the ones placed in the Letter and its Supplemental Material.
Regenerating a figure will not reproduce the shipped PDF byte for byte: a second
`savefig` under `constrained_layout` re-solves the layout, and the PDF carries a
creation timestamp. Compare rendered content, not bytes.

## Letter figures

| Item | Driver script | Outputs |
|------|---------------|---------|
| Fig. 1 existence: \|X3\| over (Ω, β₁) at N_H = 3 and N_H = 7, and the homotopy branch | `scripts/make_k3_existence_figure.py` (reads `data/two_dof_k3_fifth_harmonic_homotopy.csv` from `scripts/run_k3_fifth_harmonic_homotopy.py`) | `figures/k3_existence.{pdf,png}`, cache `data/two_dof_k3_existence_maps.npz` (delete to force a full re-solve) |
| Fig. 2 mechanism: source-vector closure and source hierarchy | `scripts/render_k3_mechanism_figure.py` (needs `data/two_dof_k3_common_damping.csv` from `scripts/run_k3_common_damping_continuation.py`) | `figures/k3_mechanism.{pdf,png}` |

The Letter carries no tables. `scripts/run_k3_main_figures.py` still renders
`figures/k3_zero_map.{pdf,png}` and `figures/k3_argand.{pdf,png}`; neither is
used in the present version of the Letter, and both are kept for provenance.

## Letter End Matter

| Item | Driver script | Outputs |
|------|---------------|---------|
| Leading-order cancellation, Eq. (EM2) | `scripts/run_k3_analytic_cancellation.py` | `data/two_dof_k3_analytic_cancellation.csv` |
| Phase reduction and closure criterion, Eqs. (EM3)–(EM5) | `scripts/run_k3_phase_reduction.py` | `data/two_dof_k3_phase_reduction{,_table,_scaling}.csv` |
| One-pathway limit, Eq. (EM6); source bound | `scripts/run_k3_gate_certificates.py` | `stdout` |
| Transversality, continuation pair (Ω, β₁) | `scripts/run_k3_ode_shooting.py` | `data/two_dof_k3_ode_shooting.csv` |
| Transversality, drive pair (Ω, F) | `scripts/run_k3_ode_shooting_drive_pair.py` | `data/two_dof_k3_ode_shooting_drive_pair.csv` |
| Finite-amplitude closure, raw and projected residuals | `scripts/run_k3_residual_decomposition.py` | `stdout` |
| Fold curve κ\*(F) and the forcing threshold | `scripts/run_k3_fold_F_free.py` | `data/two_dof_k3_fold_F_free.csv`, `data/two_dof_k3_fold_F_free_checks.csv` |
| Local constitutive validity; tangent-stiffness and snap-through bounds | `scripts/run_k3_gate_certificates.py` | `stdout` |
| Electrostatic gate path and its control Jacobian | `scripts/run_k3_gate_path.py` | `stdout` |
| Dimensional example (Table S16 values) | `scripts/run_k3_suppression.py` | `stdout` |

## Supplemental Material figures

| Item | Driver script | Outputs |
|------|---------------|---------|
| Fig. S1 two-pathway cancellation | `scripts/run_k3_two_pathway.py` | `figures/k3_two_pathway_interference_col.{pdf,png}`, `data/two_dof_k3_two_pathway.csv` |
| Fig. S2 residual decomposition and tested perturbations | `scripts/render_k3_persistence_figure.py` (renderer), `scripts/run_k3_residual_decomposition.py` (values) | `figures/k3_persistence_residual.{pdf,png}`; decomposition to `stdout` |
| Fig. S3 common-damping continuation to high Q | `scripts/render_k3_mechanism_figure.py` (renderer; needs `data/two_dof_k3_common_damping.csv` from `scripts/run_k3_common_damping_continuation.py`) | `figures/k3_high_q.{pdf,png}` |
| Fig. S4 fold of the root locus in the coupling, and the square-root fit | `scripts/run_k3_nondegenerate_exact_zero.py` | `figures/k3_nondeg_exact_zero_fold.{pdf,png}`, `data/two_dof_k3_nondeg_{status,saddlenode,locus,sqrt_windows,convergence}.csv` |
| Fig. S5(a) branch against forcing | `scripts/make_k3_branch_vs_F.py` (reads `data/two_dof_k3_weak_drive_path_kappa010.csv` and `data/k3_sister_family.csv`) | `figures/k3_branch_vs_F.{pdf,png}` |
| Fig. S5(b) notch versus zero along a frequency sweep | `scripts/run_k3_notch_vs_zero.py` | `figures/k3_notch_vs_zero.{pdf,png}`, `data/two_dof_k3_notch_vs_zero.csv` |
| Fig. S6 local sensitivity, 20 dB/decade law, Floquet along F ∈ [0.26, 0.35] | `scripts/run_k3_sensitivity.py` | `figures/k3_sensitivity.{pdf,png}` |
| Fig. S7 second oscillator model (inertial coupling) | `scripts/run_k3_rlc_second_example.py` | `figures/rlc_argand_x3.{pdf,png}`, `figures/rlc_zero_map.{pdf,png}` |

## Supplemental Material tables

| Item | Driver script | Outputs |
|------|---------------|---------|
| Table S1 truncation convergence (Sec. S2) | `scripts/run_k3_nondegenerate_exact_zero.py` | `data/two_dof_k3_nondeg_convergence.csv` |
| Table S2 1:3:5 source truncation (Sec. S2) | `scripts/run_k3_phase_reduction.py` | `data/two_dof_k3_phase_reduction_table.csv` |
| Table S3 the root against the 5:1 condition (Sec. S4) | `scripts/run_k3_modal_resonance.py` | `data/two_dof_k3_modal_resonance.csv` |
| Table S4 the zero under the exact electrostatic law (Sec. S6) | `scripts/run_k3_electrostatic_coupling.py` | `data/two_dof_k3_electrostatic_{design,roots,bias}.csv` |
| Table S5 continuation to high Q with both loss rates scaled (Sec. S6) | `scripts/run_k3_common_damping_continuation.py` | `data/two_dof_k3_common_damping{,_truncation}.csv` |
| Table S6 stability and control on the exact-law orbit at high Q (Sec. S6) | `scripts/run_k3_common_damping_continuation.py` | `data/two_dof_k3_common_damping_electrostatic.csv` |
| Table S7 absorber-only stress test (Sec. S6) | `scripts/run_k3_high_q_continuation.py` | `data/two_dof_k3_high_q_{continuation,truncation,tail}.csv` |
| Table S8 quintic fold as the forcing is varied (Sec. S6) | `scripts/run_k3_quintic_fold_F.py` | `data/two_dof_k3_quintic_fold_F.csv` |
| Table S9 quintic-regularized roots with Floquet spectra (Sec. S6) | `scripts/run_k3_quintic_regularized_root.py` | `data/two_dof_k3_quintic_regularized.csv` |
| Table S10 square-root fits about the fold apex (Sec. S7) | `scripts/run_k3_nondegenerate_exact_zero.py` | `data/two_dof_k3_nondeg_sqrt_windows.csv` |
| Table S11 fold curve κ\*(F) (Sec. S7) | `scripts/run_k3_fold_F_free.py` | `data/two_dof_k3_fold_F_free.csv` |
| Table S12 tracked zeros beyond the F = 0.30 coupling fold (Sec. S7) | `scripts/run_k3_fold_F_free.py` | `data/two_dof_k3_fold_F_free_checks.csv` |
| Table S13 minimum drive F_min against the dissipation bound, and the locked amplitude (Sec. S7) | `scripts/run_k3_weak_drive_energy_bound.py` | `data/two_dof_k3_weak_drive_energy_bound.csv` |
| Table S14 independent multi-method verification (Sec. S10) | `scripts/run_k3_multimethod_verification.py` (rows 1–3), `scripts/run_k3_nondegenerate_exact_zero.py` (row 4) | `data/two_dof_k3_multimethod{,_summary}.csv`, `data/two_dof_k3_nondeg_status.csv` |
| Table S15 perturbative comparison branch (Sec. S11) | `scripts/run_k3_sister_family.py` | `data/k3_sister_family.csv` |
| Table S16 dimensional scaling (Sec. S12) | `scripts/run_k3_suppression.py` | `stdout` |

The unnumbered tabulars in the Supplemental Material, set without float
numbers, are backed by

| Item | Driver script | Outputs |
|------|---------------|---------|
| Quintic continuation, both arms (Sec. S6) | `scripts/run_k3_extended_model_persistence.py` | `stdout` |
| Two-branch contrast (Sec. S11) | `scripts/run_k3_sister_family.py`, `scripts/run_k3_nondegenerate_exact_zero.py` | `data/k3_sister_family.csv`, `data/two_dof_k3_nondeg_convergence.csv` |
| Homotopy fold at λ\* = 0.4837, and its truncation independence at N_H = 7, 9, 11 | `scripts/run_k3_fifth_harmonic_homotopy.py`, `scripts/run_k3_homotopy_verification.py` | `data/two_dof_k3_fifth_harmonic_homotopy.csv`; `stdout` |
| Low-frequency sheet: boundary degeneracy as Ω → 0 (Sec. S7) | `scripts/run_k3_low_frequency_sheet.py` | `data/two_dof_k3_low_frequency_sheet.csv` |
| Mechanism in the second model: λ\* = 0.2925 at the selected point, no fold at the Ω = 0.228 control (Sec. S14) | `scripts/run_k3_second_model_mechanism.py` | `stdout` |

## Supplemental Material sections

| Item | Driver script | Outputs |
|------|---------------|---------|
| Sec. S1 solver, working point, σ_min and cond J_aug | `scripts/run_k3_nondegenerate_exact_zero.py` | `data/two_dof_k3_nondeg_status.csv` |
| Sec. S2 onset of the root at the fifth harmonic | `scripts/run_k3_nondegenerate_exact_zero.py`, `scripts/run_k3_residual_decomposition.py` | `data/two_dof_k3_nondeg_convergence.csv`; `stdout` |
| Sec. S2 channel geometry of the projected combination | `scripts/run_k3_phase_reduction.py` | `stdout` |
| Sec. S2 minimum \|X3\| on the 67×67 scan at N_H = 3 | `scripts/run_k3_multimethod_verification.py` | `stdout` |
| Sec. S4 modal structure, propagator gains, resonance width, 5:1 tracking | `scripts/run_k3_modal_resonance.py` | `data/two_dof_k3_modal_resonance.csv` |
| Sec. S5 phase reduction and closure criterion | `scripts/run_k3_phase_reduction.py` | `data/two_dof_k3_phase_reduction{,_table,_scaling}.csv` |
| Sec. S6 absorber detuning, absorber loss, mass ratio, quintic | `scripts/run_k3_persistence_renull.py`, `scripts/run_k3_extended_model_persistence.py` | `stdout` |
| Sec. S6 electrostatic gate path | `scripts/run_k3_gate_path.py` | `stdout` |
| Sec. S6 electrostatic route: expansion, pull-in margin, exact-law roots, (Ω, V) control pair | `scripts/run_k3_electrostatic_coupling.py` | `data/two_dof_k3_electrostatic_{design,roots,bias}.csv` |
| Sec. S6 common-damping continuation: production solver, truncation, Floquet, control Jacobian, scaled phases, and the exact-law orbit's stability and control at both ends | `scripts/run_k3_common_damping_continuation.py` | `data/two_dof_k3_common_damping{,_truncation,_electrostatic}.csv` |
| Sec. S6 absorber-only stress test, truncation checks, Floquet, detuning, mechanism split, and the tail past the 3:1 crossing | `scripts/run_k3_high_q_continuation.py` | `data/two_dof_k3_high_q_{continuation,truncation,tail}.csv` |
| Sec. S5 wide-window scan of the leading-order condition, and its scaling with ζ₂ | `scripts/run_k3_leading_order_scan.py` | `data/two_dof_k3_leading_order_{scan,crossings}.csv` |
| Sec. S6 mean-mode corrections (α₁ point; gate path with constant force) | `scripts/run_k3_mean_mode_corrections.py` | `data/two_dof_k3_mean_mode_{alpha1,gate}.csv` |
| Sec. S7 orbit stays hyperbolic through the merger (Floquet along both fold arms and at the apex) | `scripts/run_k3_fold_floquet.py` | `data/two_dof_k3_fold_floquet.csv` |
| Sec. S7 opposite-index arms, winding numbers, augmented apex | `scripts/run_k3_winding_certificate.py`, `scripts/run_k3_nondegenerate_exact_zero.py` | `stdout`; `data/two_dof_k3_nondeg_{saddlenode,status}.csv` |
| Sec. S7 two arms in the search window, with control Jacobians | `scripts/run_k3_multimethod_verification.py` | `data/two_dof_k3_multimethod.csv` |
| Sec. S7 A the fold is a curve κ\*(F) | `scripts/run_k3_fold_F_free.py` | `data/two_dof_k3_fold_F_free{,_checks}.csv` |
| Sec. S7 B route toward weak drive | `scripts/run_k3_weak_drive_route.py` | `data/two_dof_k3_weak_drive_route.csv`, `data/two_dof_k3_weak_drive_path_kappa010.csv` |
| Sec. S8 time-domain cross-check (DOP853, cold start) | `scripts/run_k3_nondegenerate_exact_zero.py` | `data/two_dof_k3_nondeg_status.csv`, fields `A3_dop853`, `dop_periodicity_resid` |
| Sec. S8 A continuous-time shooting solve, (Ω, β₁) | `scripts/run_k3_ode_shooting.py` | `data/two_dof_k3_ode_shooting.csv` |
| Sec. S8 A continuous-time shooting solve, drive pair (Ω, F) | `scripts/run_k3_ode_shooting_drive_pair.py` | `data/two_dof_k3_ode_shooting_drive_pair.csv` |
| Sec. S9 Floquet spectrum, spectral radius, Abel–Liouville check | `scripts/run_k3_quintic_regularized_root.py` (k₅ = 0 row), `scripts/run_k3_ode_shooting.py` | `data/two_dof_k3_quintic_regularized.csv`, field `rho`; `stdout` |
| Sec. S12 dimensionalization and suppression | `scripts/run_k3_suppression.py` | `stdout` |
| Sec. S13 one-pathway source zero not reached | `scripts/run_k3_gate_certificates.py` | `stdout` |
| Sec. S15 computational environment | — | `pyproject.toml`, `uv.lock` |
| Sec. S14 second oscillator model with inertial coupling | `scripts/run_k3_rlc_second_example.py` | `figures/rlc_argand_x3.{pdf,png}`, `figures/rlc_zero_map.{pdf,png}` |
| Sec. S3 homotopy in the fifth-harmonic return: fold at λ\* = 0.4837, pseudo-arclength continuation with a nullspace tangent | `scripts/run_k3_fifth_harmonic_homotopy.py` | `data/two_dof_k3_fifth_harmonic_homotopy.csv` |
| Sec. S3 homotopy verification: composition of the C₃⁽⁵⁾ coefficient, generic saddle-node conditions, λ\* at N_H = 7, 9, 11 | `scripts/run_k3_homotopy_verification.py` | `stdout` |
| Sec. S7 minimum drive against the exact periodic energy balance; effective coupling and the locked amplitude | `scripts/run_k3_weak_drive_energy_bound.py` | `data/two_dof_k3_weak_drive_energy_bound.csv` |
| Sec. S7 the low-frequency sheet and its boundary degeneracy | `scripts/run_k3_low_frequency_sheet.py` | `data/two_dof_k3_low_frequency_sheet.csv` |
| Sec. S14 the mechanism in the second model, with the Ω = 0.228 negative control | `scripts/run_k3_second_model_mechanism.py` | `stdout` |



## Notes on interpretation

- `|X_3|` at a reported root is the **constrained-solver residual at
  termination**, not a physical null depth. It rescales with the
  nondimensionalization.
- Condition numbers and Jacobian determinants are reported in the nondimensional
  coordinates defined in Sec. S1 and change under rescaling.
- Searches reported as "no root found" are bounded by the stated window and grid
  resolution, and are over zero-mean period-`T` solutions.
- Winding numbers are computed on sampled loops, not interval enclosures, so
  they are numerical evidence for the local degree of the discretized map rather
  than validated proofs.
- The default harmonic-balance basis carries `k = 1..N_H` with no mean mode.
  That is exact wherever `alpha_j = 0`, because the orbit is half-period
  antisymmetric and both mean equations are then satisfied identically. Any
  calculation with `alpha_j != 0` must pass `include_mean=True` (and, for the
  gate path, the constant electrostatic force via `force_dc`), which appends
  `a0^(1), a0^(2)` after the oscillatory coefficients and enforces the two
  zero-frequency balances. Existing `coefficient_index` values are unchanged.
- The deposit is the record: it is self-contained, and `uv.lock` pins the
  environment. Versions are distinguished by the Zenodo DOI rather than by a
  commit hash of the working repository.
