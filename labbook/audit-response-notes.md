# Audit responses — local evidence and reply drafts

Prepared 2026-08-27. **Not posted to GitHub.** These are proposed modeling-team
responses, not changes to the auditors' issue status.

Formal audit target: `1f1899f80c65ad2523f153a8eacfd51eeccc449d`.
Evidence commit in `mindrl-challenge/submission-wl`:
`b99c7a72a4d50c6dfdf61afb1f9102b0617d537c`.
Later model changes do not replace that audit target.

## Q6 — equivalent parameterization

[Target issue #2](https://github.com/mindrl-challenge/submission-the-brain-crackers/issues/2)
and [pinned comparison](https://github.com/mindrl-challenge/submission-wl/blob/b99c7a72a4d50c6dfdf61afb1f9102b0617d537c/audits/the_brain_crackers/examining-two-stage-theory/outputs/comparison.json).

Let `l` be the last action, `g` the gate logit, and `u` the base utilities.
Set the matched single-softmax utilities to `v_a=u_a` for `a != l` and
`v_l=g+log(sum(exp(u_j), j != l))`. Its probability for `l` is `sigmoid(g)`;
the other probabilities equal `(1-sigmoid(g))*softmax(u_nonlast)`.
Applying the same uniform lapse preserves equality.

The auditors' matched model changes validation NLL by only -2.86e-10 relative
to the two-stage form; validation accuracy is identical. The restricted
constant-repeat model has validation NLL 0.610540 versus 0.534969, a difference
of 0.075571. That restriction also removes dynamic predictors and changes
effective flexibility. It cannot isolate a psychological benefit of two stages.

Our independent numerical test, `tests/test_audit_equivalence.py`, checks the
identity across synthetic histories and 2, 4 and 7 actions. We inspected the
pinned aggregate comparison; we did **not** rerun the auditors' full grid.

### Reply draft for Q6

Thank you for the matched control and replication. We agree that the submitted
stay/switch rule is exactly reparameterizable as a single softmax with a dynamic
repeat utility. We have independently checked that identity numerically. The
restricted shared-rule comparison supports the usefulness of the retained
dynamic predictors under the recorded development procedure, not a uniquely
two-stage psychological process. We have narrowed the interpretation card
accordingly and retain the frozen SHA as the formal audit target. We propose
`scope-narrowed` as the resolution, subject to the auditing team's agreement.
The post-freeze documentation commit will be linked when published.

## Q7 — same-model parameter recovery

[Target issue #3](https://github.com/mindrl-challenge/submission-the-brain-crackers/issues/3)
and [pinned recovery report](https://github.com/mindrl-challenge/submission-wl/blob/b99c7a72a4d50c6dfdf61afb1f9102b0617d537c/audits/the_brain_crackers/testing-parameter-recovery/outputs/recovery_analysis/recovery_analysis.json).

We independently recalculated correlations and MSE from the 100 rows in the
auditors' `outputs/model_fits/fit_manifest.csv`, and checked all generating
values against `outputs/generating_parameters.csv` at the evidence commit.
The result is recorded in `artifacts/audit_evidence_verification.json`.
`experiments/verify_audit.py` reproduces the arithmetic from an authorized audit
checkout. No auditor implementation was imported into our model.

| Parameter | Pearson r (100 parameter sets) |
|---|---:|
| Value sensitivity | 0.993480 |
| Uncertainty sensitivity | 0.984405 |
| Stay intercept | 0.974457 |
| Stay value weight | 0.988409 |
| Linear surprise | 0.991983 |
| Squared surprise | 0.987794 |
| Run length | 0.953998 |
| Trial index | 0.990429 |
| Lapse | 0.812685 |

The primary evidence has 99 clean fits and one fit with an optimizer warning.
Each simulated dataset contains 100 trajectories, 98 participants and 11,957
trials, not the full human public-data sample. The noise variances were fixed
at q=25 and r=200. The separate 700-fit sample-size series uses 1, 2, 4, 8, 16,
32 and 64 participants. We did not regenerate the simulations or repeat these
800 fits. Slightly different rounded correlations in the issue's Windows
cross-replication are not new independent datasets.

### Reply draft for Q7

Thank you for the recovery and sample-size analyses. We independently
recalculated the primary 100-fit correlations from the pinned estimates and
confirmed the generating values. Eight coefficients recover strongly in this
same-model experiment (r approximately 0.954–0.993); lapse is weaker
(r approximately 0.813), and one fit has an optimizer warning. We now state
explicitly that this is recovery under a correctly specified generating model,
the tested parameter ranges and fixed Kalman variances. It does not establish
unique psychological meaning, identification of the noise variances, robustness
to model misspecification, or recovery of any post-freeze added coefficients.
We propose `acknowledged` as the resolution, subject to the auditing team's
agreement. The corresponding documentation commit will be linked when published.

## Publication checklist

- Review the wording as a team, then commit the final package.
- Add the exact published documentation/model SHA to each reply before posting.
- Post to the existing issues; do not create duplicate findings.
- Ask the auditors to confirm the resolution; do not silently change their
  formal target or claim they reviewed the final model.
