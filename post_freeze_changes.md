# Post-freeze changes

## Version identification

| Version | Immutable identification |
|---|---|
| Formal Phase-1 audit target | `1f1899f80c65ad2523f153a8eacfd51eeccc449d` |
| Optional post-freeze checkpoint | None requested in this development pass. No separate optional-checkpoint SHA is recorded in this package; the team must add it if it made another request. |
| Final package | The immutable commit containing this file, pinned by the team's final rolling-submission issue and central registry entry. Resolve its exact SHA with `git rev-parse HEAD` after checkout. |

A commit cannot embed its own containing SHA. The final issue/registry is the
authoritative 40-character pin, not a self-referencing field in `submission.yaml`.
The final parameter artifact is `artifacts/fitted_params_final.json`, SHA-256
`62dfebd8febd7113c0d69fd7fe5b5e4d72ac382a06c3b1423ea4a1538e25fded`.
This distinguishes the final content even before the team publishes its commit.
The original parameter artifact remains unchanged (SHA-256
`7a374bdc31549e6a70b18e8062712e34c7f377c87d3498d7e1f14376c2d32a01`).

The original accepted submission is documented in
[issue #63](https://github.com/mindrl-challenge/mindrl-challenge-core/issues/63)
and [PR #64](https://github.com/mindrl-challenge/mindrl-challenge-core/pull/64).
Neither that acceptance nor the local changes automatically registers the final
package. Follow the final checklist for publication and intake.

## Model changes

- Retained the fixed-noise Kalman value/uncertainty updates, dynamic repeat
  gate, conditional alternative probabilities and uniform lapse.
- Added four fixed-coefficient history features: Beta(2,2)-smoothed past
  stay/switch log odds; a surprise trace with fixed update rate 0.2;
  centered log choice frequency; and centered negative log selection age.
- All additional state is within a trajectory and is cleared by reset.
  There is no online optimization, subject lookup or cross-trajectory state.
- Kept process variance at 25. Selected observation variance 100 instead of
  the checkpoint's 200 through the prespecified public-data comparison.
- Fitted thirteen decision coefficients offline; no ensemble was deployed.
  The original nine-coefficient behavior is available with `config_baseline.yaml`.

## Selection and verification evidence

Five subject-disjoint folds, seed 20260827, compared six fitted specifications
and a fixed equal-weight mixture. Each fold refitted its baseline and candidates
using training subjects only and two data-independent starting points. All
candidate results are retained in `artifacts/final_development_cv.json`.

| Comparison | Baseline | Final |
|---|---:|---:|
| Pooled development CV NLL | 0.569687 | 0.560396 |
| Pooled development CV accuracy | 79.6825% | 79.8775% |
| Full-public fit NLL | 0.569410 | 0.560104 |
| Full-public fit accuracy | 79.7015% | 79.8883% |

Final NLL improves in all five folds. The selection rule required lower pooled
NLL, no pooled accuracy loss and improvement in at least four folds. The
history-utility-only candidate reduced NLL but lowered pooled accuracy and was
not eligible. See the interpretation card for paired descriptive intervals and
the limits of reused development data. No hidden score selected the model.

The tested history implementation was moved into the standalone `agent.py`;
the experiment module now imports it, avoiding duplicate inference logic.
Numerical-gradient, vectorized/runtime, reset, causality, arbitrary action-label
and loader tests cover the integration. The final sequential score is recorded
in `artifacts/final_public_evaluation.json`.

## Audit handling

| Finding | Response in this package | Proposed resolution |
|---|---|---|
| [Q6 / issue #2](https://github.com/mindrl-challenge/submission-the-brain-crackers/issues/2) | Acknowledged exact single-softmax equivalence; added our numerical identity check; removed the claim that prediction establishes two psychological stages. | `scope-narrowed` |
| [Q7 / issue #3](https://github.com/mindrl-challenge/submission-the-brain-crackers/issues/3) | Independently recomputed the primary correlations/MSE; stated same-model/fixed-variance/sample-size limits and weaker lapse recovery. | `acknowledged` |

These are proposed modeling-team responses, not assertions that issue status
has already changed. Drafts are in `labbook/audit-response-notes.md` and must be
reviewed, linked to the published SHA and posted by the team. The auditors are
not assumed to have re-audited the final model. In particular, Q7 does not
establish recovery of the four additional coefficients or the final r=100 model.

## Documentation, scoring and provenance

- Focused the interpretation card on the submitted model, evidence and limits;
  moved legacy evolution and the previously reported adaptive-noise negative
  result to the labbook, clearly marked as not rerun in this pass.
- Kept GitHub-compatible math delimiters and removed diagram/math code fences
  and unsupported macros. Local MathJax preview is not an online GitHub/Typora
  certification; the team should check the published view.
- Corrected local accuracy ties to use exact equality, matching the official
  metric. This did not change the frozen model's reproduced headline score.
- Added a post-lapse switch-occurrence/target loss decomposition and input,
  configuration and parameter hashes. No raw trajectories are committed.
- Public scoreboard summaries informed the initial gap assessment; no private
  competing implementation was imported or mined for this model.

Workflow reference: [Auditing operations](https://github.com/mindrl-challenge/mindrl-challenge-core/blob/main/docs/auditing_operations.md).
