# Interpretation Card — History-Augmented Stay-Switch Kalman

## 1. Submission and claim

This choice-only model predicts the next human action from previously revealed
actions and rewards. It combines fixed-noise Kalman value beliefs with
trajectory-local reward and choice history. A stay gate and a conditional
distribution over alternatives make the probability calculation inspectable.

The claim is predictive and descriptive. The factorization is **not evidence
for two separate psychological stages**: an exactly matched single softmax
produces the same probabilities (audit Q6). Parameter names do not establish
unique psychological causes.

The standalone [agent.py](agent.py) uses only the Python standard library.
There is no neural network, external API, participant lookup, future observation
or optimizer step at evaluation time. Global coefficients are read at
construction from [the final parameter artifact](artifacts/fitted_params_final.json).

## 2. Mechanism map

| Component | Implementation and interpretation boundary |
|---|---|
| Value learning | A Gaussian mean and variance for each action; reward updates use a Kalman gain. The noise variances are fixed assumptions. |
| Value and uncertainty | Relative value and standard deviation contribute to action utilities. This does not uniquely identify directed exploration. |
| Choice-history utilities | Centered log frequency and negative log age since selection affect both switch-target ranking and the gate's base log-odds. |
| Dynamic persistence | Last signed/squared surprise, run length, trial count, smoothed past stay frequency and a surprise trace enter the gate. |
| Lapse | A uniform probability mixture; it may absorb unmodeled behavior rather than a distinct psychological lapse process. |
| State scope | Beliefs and all history accumulators reset for every trajectory. No state is shared across subjects or trajectories. |

## 3. Formal model

### 3.1 Beliefs and outcome updates

At decision $t$, $Q_{t,a}$ and $V_{t,a}$ are the mean and variance **before**
the current reward is assimilated. For observed action $a_t$ and reward $r_t$:

$$e_t = r_t-Q_{t,a_t}, \qquad K_t = \frac{V_{t,a_t}}{V_{t,a_t}+\sigma_{\mathrm{obs}}^2}.$$

$$Q_{t+1,a_t}=Q_{t,a_t}+K_t e_t, \qquad V^{\mathrm{post}}_{t,a_t}=(1-K_t)V_{t,a_t}.$$

Unselected means and posterior variances are unchanged. Before the next
decision, every action receives fixed process variance:

$$V_{t+1,a}=\min\left(V^{\mathrm{post}}_{t,a}+\sigma_{\mathrm{proc}}^2,V_{\max}\right).$$

The standardized surprise is computed from the pre-update belief:

$$\delta_t=\frac{e_t}{\sqrt{V_{t,a_t}+\sigma_{\mathrm{obs}}^2}}.$$

Missing or non-finite rewards leave the selected mean unchanged and set
$\delta_t=0$. Variance propagation and observed-choice history updates still
occur; the surprise trace decays as specified below.

### 3.2 Value, uncertainty and choice-history utilities

For available-action set $A$, let $\bar Q_t$ and $\bar S_t$ be the across-action
means of $Q_{t,a}$ and $\sqrt{V_{t,a}}$. Define the common scale:

$$D_t=\sqrt{\frac{1}{|A|}\sum_{a\in A}\left[(Q_{t,a}-\bar Q_t)^2+V_{t,a}\right]}.$$

$$z^Q_{t,a}=\frac{Q_{t,a}-\bar Q_t}{D_t}, \qquad z^U_{t,a}=\frac{\sqrt{V_{t,a}}-\bar S_t}{D_t}.$$

Let $n_{t,a}$ count past selections of action $a$. Let $d_{t,a}$ be the number
of completed trials since its last selection: zero immediately after selection,
or $t$ if it has never been selected. Define $F_{t,a}$ by centering
$\log(1+n_{t,a})$ across actions and $C_{t,a}$ by centering
$-\log(1+d_{t,a})$ across actions. Then:

$$u_{t,a}=\beta_Q z^Q_{t,a}+\beta_U z^U_{t,a}+\beta_F F_{t,a}+\beta_C C_{t,a}.$$

$$b_{t,a}=\frac{\exp(u_{t,a})}{\sum_{j\in A}\exp(u_{t,j})}.$$

All features are computed before the current observed action and reward.

### 3.3 Stay gate

After the first trial, let $l=a_{t-1}$ be the last action, $R_t$ its current
run length, and $t$ the number of completed trials. With past stay and switch
counts $N_t^{\mathrm{stay}}$ and $N_t^{\mathrm{switch}}$, the smoothed persistence
feature and recent-surprise trace are:

$$H_t=\log\left(\frac{N_t^{\mathrm{stay}}+2}{N_t^{\mathrm{switch}}+2}\right).$$

$$E_{t+1}=0.8E_t+0.2\delta_t, \qquad E_0=0.$$

The first observed action adds no stay/switch count. These are within-trajectory
history summaries, not online fitting of global coefficients.

$$\begin{aligned} g_t={}&b_0+b_Q\,\mathrm{logit}(b_{t,l})+b_{\delta}\delta_{t-1}+b_{\delta^2}\delta_{t-1}^2\\ &+b_R\log(1+R_t)+b_T\log(1+t)+b_H H_t+b_E E_t. \end{aligned}$$

$$s_t=\frac{1}{1+\exp(-g_t)}.$$

The gate-only terms alter persistence without changing the relative ranking
among alternatives. Frequency and recency utilities also enter the base
log-odds, so their effects are not confined to switch targets.

### 3.4 Returned probabilities

For $a\ne l$, define the conditional alternative distribution:

$$c_{t,a}=\frac{\exp(u_{t,a})}{\sum_{j\in A\setminus\{l\}}\exp(u_{t,j})}.$$

$$\tilde p_{t,a}=\begin{cases}s_t,&a=l,\\(1-s_t)c_{t,a},&a\ne l.\end{cases}$$

$$p_{t,a}=(1-\lambda)\tilde p_{t,a}+\frac{\lambda}{|A|}.$$

The first prediction is uniform: all beliefs and history features are initially
symmetric, and the model does not use other pre-choice metadata. A one-action
context returns probability one. For numerical safety, variances and $D_t$ are
floored at $10^{-12}$, base probabilities entering the logit are clipped to
$[10^{-9},1-10^{-9}]$, and returned probabilities are normalized.

## 4. Final fixed parameters

Thirteen decision coefficients are fitted offline. Initial beliefs, q=25 and
the variance cap are fixed; r=100 was selected from the limited development
comparison with r=200. Full precision is stored in the parameter artifact.

| Parameter | Value | Operational role |
|---|---:|---|
| `initial_value` | 50.000000 | Initial reward mean |
| `initial_variance` | 400.000000 | Initial belief variance |
| `process_variance` | 25.000000 | Fixed per-trial variance increment |
| `observation_variance` | 100.000000 | Fixed observation variance |
| `variance_ceiling` | 10000.000000 | Variance cap |
| `value_sensitivity` | 1.086762 | Value utility coefficient |
| `uncertainty_sensitivity` | 1.000137 | Relative-uncertainty utility coefficient |
| `stay_intercept` | -0.052226 | Gate intercept |
| `stay_value_weight` | 1.640779 | Previous-action base log-odds |
| `surprise_linear_weight` | 0.915180 | Last signed surprise |
| `surprise_quadratic_weight` | -0.728861 | Last squared surprise |
| `run_length_weight` | 1.129567 | Current run length |
| `trial_index_weight` | 0.079031 | Completed trial count |
| `persistence_trace_weight` | 0.501821 | Smoothed past persistence |
| `surprise_trace_weight` | 0.871108 | Recent surprise trace |
| `frequency_weight` | 0.167488 | Action selection frequency |
| `recency_weight` | 0.089782 | Action selection recency |
| `lapse` | 0.014358 | Uniform mixture |

Positive coefficients describe associations conditional on the other terms.
The uncertainty coefficient is not, by itself, evidence of a uniquely
identified exploration process. History terms can trade off with run length,
time and uncertainty; recovery of the added coefficients has not been tested.

## 5. Public-data development evidence

The public input contains 2,678 trajectories, 689 subjects and 320,080 trials.
The [executed data-check notebook](analysis/final_data_checks.ipynb) verifies
IDs, duplicates, rewards, actions and metric denominators. No records were
excluded. The input hash is recorded with the results.

The [prespecified development protocol](labbook/2026-08-27-final-development.md)
uses five subject-disjoint folds, seed 20260827, and identical trial weights
for every candidate. Each fold fits on its training subjects only, with two
fixed starts, bounded L-BFGS-B and at most 250 iterations per start. The
all-public fitted parameters are never used as fold initialization. Selection
requires lower pooled NLL, no pooled accuracy loss and NLL improvement in at
least four folds. The candidate list and all fold results, including
unsuccessful alternatives, are in [the complete CV record](artifacts/final_development_cv.json).

| Development candidate | Pooled out-of-fold NLL | Accuracy |
|---|---:|---:|
| Refitted baseline | 0.569687 | 79.6825% |
| History gate only | 0.564034 | 79.7908% |
| History utility only | 0.567087 | 79.6617% |
| Combined history, r=200 | 0.561184 | 79.7821% |
| Combined history, stronger L2 | 0.562238 | 79.7555% |
| Combined history, r=100 (selected) | 0.560396 | 79.8775% |
| Fixed 50:50 mixture | 0.563200 | 79.7921% |

The selected candidate improves NLL in all five folds. Relative to the refitted
baseline, pooled NLL changes by -0.009291 and accuracy by +0.1950 percentage
points. Paired subject-bootstrap 95% intervals are [-0.010793, -0.007894] for
NLL and [+0.1015, +0.2928] percentage points for accuracy (2,000 resamples).

These are descriptive intervals conditional on the fitted out-of-fold
predictions, not selection-adjusted tests or intervals covering training
uncertainty. This public dataset and the baseline have already been explored;
the CV was itself used to select the final candidate. It is not an independent
final test or a guarantee of hidden-task improvement.

After selection, the model was refitted on all public data and frozen:

| Full-public fit diagnostic | Mean NLL | Accuracy |
|---|---:|---:|
| Preserved checkpoint artifact | 0.569410 | 79.7015% |
| Final artifact, sequential runtime evaluation | 0.560104 | 79.8883% |

The [final sequential evaluation](artifacts/final_public_evaluation.json) uses
prediction before update, natural-log NLL, and fractional accuracy credit for
**exact** tied maxima, matching the official convention. The full-public
scores are in-sample diagnostics, not generalization estimates.

| Trial category | Trials | Final mean NLL |
|---|---:|---:|
| First | 2,678 | 1.386294 |
| Stay | 239,937 | 0.178333 |
| Switch | 77,465 | 1.714023 |

Using final post-lapse probabilities, switch NLL decomposes into 0.757820 for
switch occurrence and 0.956202 for the conditional target. Both components
matter; the full switch loss must not be attributed to target choice alone.
Uniform four-action prediction has NLL 1.386294 and fractional accuracy 25%.

## 6. Audits and interpretation boundaries

Both formal audits concern checkpoint
**1f1899f80c65ad2523f153a8eacfd51eeccc449d**, not the post-freeze model.

**Q6 — equivalent parameterization.** Keep $v_{t,a}=u_{t,a}$ for $a\ne l$ and set:

$$v_{t,l}=g_t+\log\left(\sum_{a\ne l}\exp(u_{t,a})\right).$$

A single softmax over $v$ gives exactly the same probabilities before lapse;
the same lapse preserves equality. The auditors' matched representations both
have validation NLL 0.534969. Their restricted constant-repeat control scores
0.610540, but also removes dynamic predictors, so it cannot isolate a
psychological benefit of two stages. We independently check the identity
numerically; we did not repeat the full audit grid.
See [issue #2](https://github.com/mindrl-challenge/submission-the-brain-crackers/issues/2)
and [pinned evidence](https://github.com/mindrl-challenge/submission-wl/blob/b99c7a72a4d50c6dfdf61afb1f9102b0617d537c/audits/the_brain_crackers/examining-two-stage-theory/outputs/comparison.json).

**Q7 — conditional recovery.** Across 100 synthetic datasets (each 100
trajectories, 98 participants and 11,957 trials), eight frozen-model
coefficients have Pearson r approximately 0.954–0.993; lapse is weaker at
0.813. There are 99 clean fits and one optimizer warning. We independently
recalculated correlations and MSE from the pinned estimates and verified their
generating values, without regenerating simulations or repeating the 800
full/subset fits. This is same-model recovery with q=25 and r=200 fixed, over
the tested parameter ranges. It neither identifies noise variances nor
establishes recovery for the four added coefficients or the final r=100 model.
See [issue #3](https://github.com/mindrl-challenge/submission-the-brain-crackers/issues/3),
[pinned evidence](https://github.com/mindrl-challenge/submission-wl/blob/b99c7a72a4d50c6dfdf61afb1f9102b0617d537c/audits/the_brain_crackers/testing-parameter-recovery/outputs/recovery_analysis/recovery_analysis.json)
and [our arithmetic check](artifacts/audit_evidence_verification.json).

[Response notes](labbook/audit-response-notes.md) preserve the local reply
drafts. [Post-freeze changes](post_freeze_changes.md) distinguish the audit
target and final package; no re-audit of the final model is implied.

## 7. Limitations and next discriminative tests

- Fixed random-walk beliefs and an initial reward mean of 50 may transfer poorly
  to different reward scales, structured dynamics, action counts or tasks.
- A global coefficient set with local history does not identify stable
  participant types. State is deliberately not carried between trajectories.
- The history additions improve this development comparison but are not a
  mechanism-specific psychological experiment. Fit reward-free history,
  fixed-learning-rate and feature-ablation controls on independent data before
  stronger claims.
- Noise variances are selected offline; the model does not claim to separate
  volatility and observation noise online.
- Parameter recovery under misspecification, and recovery for the final
  thirteen-coefficient specification, remain untested.
- No response-time prediction or hidden evaluation was performed in this pass.

## 8. Reproducibility and confidentiality

Runtime configuration: [config.yaml](config.yaml). Preserved reference:
[config_baseline.yaml](config_baseline.yaml) and the unchanged
[checkpoint parameters](artifacts/fitted_params.json). No raw trajectories,
hidden scores, private source mappings or credentials are included in the
parameter/report artifacts.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python evaluate_public.py
python evaluate_public.py --config config_baseline.yaml
```

The six offline numerical tests require the fitting dependencies; without
them they are explicitly skipped. Full development reproduction writes new
outputs rather than replacing the deployed parameters:

```powershell
python -m pip install -r requirements-fit.txt
python -m experiments.final_candidates --workers 4 --refit --output artifacts/reproduced_cv.json --parameter-output artifacts/refitted_candidate.json
```

The [historical labbook](labbook/historical-development.md) retains earlier
negative results with provenance limitations. [The final checklist](FINAL_SUBMISSION_CHECKLIST.md)
covers team review, exact commit pinning, audit replies and formal intake.
