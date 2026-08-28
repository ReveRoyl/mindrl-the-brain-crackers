# Small model and evidence artifacts

| File | Role |
|---|---|
| `fitted_params_final.json` | Deployed final parameters selected by development CV. |
| `fitted_params.json` | Preserved Phase-1 parameter artifact; do not overwrite. |
| `final_development_cv.json` | Five folds × six fitted candidates, fixed mixture, convergence records, selection and paired bootstrap. |
| `final_public_evaluation.json` | Full sequential runtime evaluation, elapsed time and source/config/parameter hashes. |
| `baseline_public_evaluation.json` | Frozen reference score reproduced before changes. |
| `audit_evidence_verification.json` | Independently recomputed Q7 arithmetic and identified Q6 comparison evidence. |
| `public_data_quality.json` | Aggregate input-integrity checks from the executed notebook. |
| `final_validation.json` | Final runtime/vectorized parity and package checks. |

All scores here use public development data or identified synthetic audit
evidence. They are not official hidden-evaluation results. No raw participant
records or other teams' implementation code are included.

Final fitting: `python -m experiments.final_candidates`.
Final verification: `python -m experiments.validate_final`.
Reproduction commands and interpretation caveats are in the repository README.
