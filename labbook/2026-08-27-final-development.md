# Final development protocol — 2026-08-27

## Scope fixed before the new experiment results

Frozen reference: `1f1899f80c65ad2523f153a8eacfd51eeccc449d`.
Keep `artifacts/fitted_params.json` unchanged. All training uses only
`data/public_train.jsonl`; no hidden scores, reward-schedule sidecars, participant
identity features, or other teams' implementation code are used for selection.

- Five subject-disjoint folds, seed 20260827. Every trajectory from a subject
  stays in one fold. Each comparison uses the identical folds and trial weights.
- Refit the nine-parameter baseline inside each training fold. Never initialize
  from the all-public fitted artifact. Use two fixed, data-independent starts;
  choose the converged solution with lowest penalized training loss.
- Fixed Kalman variances: q=25, r=200. Initial beliefs and variance cap unchanged.
- Candidates: baseline; history gate (smoothed past stay frequency and a surprise
  trace); history targets (action frequency and recency); combined history;
  combined history with stronger L2; combined history with r=100.
- Fixed history definitions: Beta(2,2)-smoothed within-trajectory stay frequency
  in log odds; surprise EMA with rate 0.2; centered log(1+choice count) and
  negative centered log(1+trials since last choice), including a never-chosen
  action's age since trajectory start. All features are computed BEFORE the
  observed action/reward, with complete reset between trajectories.
- L2=1e-5 for baseline/gate terms, including added gate coefficients;
  added utility coefficients also regularized. Strong-L2 candidate uses 1e-3.
  L-BFGS-B, at most 250 iterations per start. Save convergence, parameters,
  fold metrics and the input hash. No repeated post-hoc search.
- Primary selection: pooled trial-weighted out-of-fold NLL. Accuracy is a
  guardrail: require no pooled accuracy loss and NLL improvement in at least
  four of five folds. Compare a fixed 50:50 baseline/combined-history mixture
  as an additional conservative candidate, without tuning mixture weights.
- Report paired subject-bootstrap intervals (seed 20260827, 2000 draws) for
  selected-vs-baseline NLL and fractional accuracy. These are descriptive
  development uncertainty intervals, not selection-adjusted significance tests.
- If no candidate clears the guardrails, keep the frozen model. Otherwise refit
  only the selected model on all public data into a NEW parameter artifact.
  Validate vectorized/online parity, analytic gradients, causal update order,
  loader compatibility, finite probabilities and reset behavior before promotion.

These are development cross-validation results: this public dataset and the
baseline design have already been explored. They are not a new independent test
set, a hidden-test result, or a promise of leaderboard improvement. The original
80/20 development score is not directly comparable to pooled five-fold CV.

Execution note: after the first complete fold was saved, the remaining four
folds were dispatched as independent processes to reduce elapsed time. The
completed fold was resumed from its saved coefficients and its metrics checked
again. This changed scheduling only, not candidates, seeds, fitting starts,
convergence criteria or selection rules.

## Documentation and audit scope

Keep the final interpretation card focused on the submitted model. Move legacy
evolution and negative adaptive-noise results into the labbook, preserving their
provenance and limitations. Respond to Q6 by acknowledging the exact single-
softmax reparameterization; respond to Q7 with the scope of same-model recovery.
Create `post_freeze_changes.md` and local reply drafts. Do not publish replies,
push commits, or update the formal submission SHA without team confirmation.

## Outcome and verification

Selected `history_r100`: all four history features, q=25, r=100, L2=1e-5.
No extra post-hoc candidate or tuned ensemble weight was added.

| Evaluation | Baseline NLL | Final NLL | Baseline accuracy | Final accuracy |
|---|---:|---:|---:|---:|
| Pooled subject-grouped development CV | 0.569687 | 0.560396 | 79.6825% | 79.8775% |
| Full-public fit diagnostic | 0.569410 | 0.560104 | 79.7015% | 79.8883% |

The selected candidate improves both metrics in every fold. The utility-only
candidate has lower pooled NLL but lower accuracy and fails the guardrail.
All 60 CV starts and both full-public starts converged. Full results, including
all non-selected specifications, are in `artifacts/final_development_cv.json`.

`artifacts/final_validation.json` checks all 320,080 trials: maximum final
runtime/vectorized probability difference 5.55e-16; maximum reference-policy
compatibility difference zero. The preserved parameter file has its original
SHA-256. The final card's 18 fixed numerical settings match the artifact.

The full suite passed 24 tests under Python 3.11.5. Sequential final evaluation
also passed under the team's Python 3.10.20 environment. The data-quality
notebook executed top-to-bottom, and the final local MathJax preview rendered
47 expressions without parse errors or page-level horizontal overflow. These
local checks do not constitute official registry CI or a hidden evaluation.

Reproduction notes: the six numerical fitting tests need `requirements-fit.txt`.
Windows sandbox permissions were needed for Jupyter's secure connection file
and multiprocessing pipes; no package or system-setting changes were made.
The experiment helper now defaults refits to a separate candidate artifact and
rejects attempts to overwrite preserved or deployed parameter files.

Remaining publication decisions: confirm the complete team metadata and any
separately requested optional hidden checkpoint, review/post audit replies,
commit/push the intended package and register its exact SHA. These are not
automatically performed by the local development run.
