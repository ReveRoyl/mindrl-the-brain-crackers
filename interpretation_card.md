# Interpretation Card - Stay-Switch Kalman Agent

## 1. Submission summary

The Stay-Switch Kalman Agent is a cognitive reinforcement-learning model for
one-step-ahead prediction of human choices. It represents each available action
with a Gaussian belief over latent value and updates that belief after observed
rewards. Its decision rule separates two related questions:

1. Will the participant repeat the previous action?
2. If the participant switches, which alternative action will they choose?

This separation is the main change from the earlier StoVol-Lite model, which
placed value, uncertainty, and an accumulating choice trace into one
multi-action softmax. The submitted model uses no neural network, external API,
future observation, or optimizer step during evaluation.

## 2. Scope of the claim

The model makes the following predictive claim:

> Human choice persistence and switch-target selection are better represented
> as distinct but coupled decisions than as a single softmax with a shared
> perseveration term.

The public-data results support this architecture as a predictive model. They
do not establish that the fitted variables are uniquely identifiable
psychological processes, and they do not constitute a hidden-evaluation score.

## 3. Mechanism map

| Cognitive concept | Model implementation | Status relative to StoVol-Lite v0.1 |
|---|---|---|
| Latent action value | A Gaussian mean is maintained for each action and updated from its observed reward. | Retained |
| Belief uncertainty | An action-specific variance contracts after observation and propagates by fixed process variance before the next choice. | Retained, but no longer adapted by the previous heuristic |
| Value-guided choice | Standardized posterior means enter action utilities through `value_sensitivity`. | Retained |
| Directed exploration | Relative posterior standard deviations enter utilities through `uncertainty_sensitivity`. | Retained |
| Stay versus switch | A logistic gate assigns probability to repeating the previous action. | New |
| Switch-target selection | Conditional on switching, a softmax compares only the other available actions. | New |
| Reward surprise | Signed and squared standardized reward prediction error enter the stay gate. | Reformulated |
| Perseveration | Log run length enters only the stay gate. | Replaces the accumulating choice trace |
| Time-on-task trend | Log trial count allows a gradual within-trajectory change in stay tendency. | New |
| Lapse | A small uniform mixture prevents zero probabilities and represents occasional task-independent responding. | Retained |
| Adaptive stochasticity | No online parameter or noise-variance adaptation is performed. | Removed after ablation |

The fitted global parameters are loaded once from
`artifacts/fitted_params.json`. During evaluation, only trajectory-local
beliefs, previous reward surprise, run length, and trial count change.

## 4. Formal model

### 4.1 Action-value beliefs

For each available action $a$, the agent maintains a Gaussian belief with mean
$Q_{t,a}$ and variance $V_{t,a}$. When action $a_t$ produces reward $r_t$, the
innovation and Kalman gain are

```math
e_t = r_t - Q_{t,a_t},
```

```math
K_t = \frac{V_{t,a_t}}{V_{t,a_t} + \sigma_{\mathrm{obs}}^2}.
```

The selected action is updated by

```math
Q_{t+1,a_t}
=
Q_{t,a_t} + K_t e_t,
```

```math
V^{\mathrm{post}}_{t,a_t}
=
(1-K_t)V_{t,a_t}.
```

Before the next decision, every action variance receives the fixed process
variance:

```math
V_{t+1,a}
=
\min\left(
V^{\mathrm{post}}_{t,a} + \sigma_{\mathrm{proc}}^2,
V_{\max}
\right).
```

For an unselected action, its current variance serves as
$V^{\mathrm{post}}_{t,a}$. Missing or non-finite rewards do not update the
selected mean; reward surprise is set to zero, and uncertainty still
propagates.

The standardized reward surprise retained for the next choice is

```math
\delta_t
=
\frac{r_t-Q_{t,a_t}}
{\sqrt{V_{t,a_t}+\sigma_{\mathrm{obs}}^2}}.
```

All quantities on the right-hand side are evaluated before assimilating the
current reward.

### 4.2 Standardized value and uncertainty features

Let $\bar Q_t$ be the mean action value, and let $\bar S_t$ be the mean
posterior standard deviation across the $A_t$ available actions. The common
scale is

```math
D_t
=
\sqrt{
\frac{1}{|A_t|}
\sum_{a\in A_t}
\left[
(Q_{t,a}-\bar Q_t)^2 + V_{t,a}
\right]
}.
```

The value and relative-uncertainty features are

```math
z^Q_{t,a}
=
\frac{Q_{t,a}-\bar Q_t}{D_t},
```

```math
z^U_{t,a}
=
\frac{\sqrt{V_{t,a}}-\bar S_t}{D_t}.
```

The base utility and base choice distribution are

```math
u_{t,a}
=
\beta_Q z^Q_{t,a}
+
\beta_U z^U_{t,a},
```

```math
b_{t,a}
=
\frac{\exp(u_{t,a})}
{\sum_{j\in A_t}\exp(u_{t,j})}.
```

### 4.3 Stay-switch gate

After the first trial, let $a_{t-1}$ denote the previous action, $R_t$ the
current run length, and $t$ the number of completed trials. The stay-gate logit
is

```math
\begin{aligned}
g_t
=\;&
b_0
+ b_Q\,\operatorname{logit}(b_{t,a_{t-1}})
+ b_{\delta}\delta_{t-1} \\
&+ b_{\delta^2}\delta_{t-1}^2
+ b_R\log(1+R_t)
+ b_T\log(1+t).
\end{aligned}
```

The probability of repeating the previous action is

```math
p_t^{\mathrm{stay}} = \operatorname{sigmoid}(g_t).
```

This gate allows value, reward surprise, choice history, and within-trajectory
time to affect persistence without changing the ranking among switch targets.
The trial-count coefficient is a descriptive time trend; it should not be
interpreted as direct evidence of fatigue or learning without a separate test.

### 4.4 Conditional switch-target choice

If the participant switches, the conditional probability of selecting
$a\ne a_{t-1}$ is

```math
c_{t,a}
=
\frac{\exp(u_{t,a})}
{\sum_{j\in A_t\setminus\{a_{t-1}\}}\exp(u_{t,j})}.
```

Before lapse mixing, the action probabilities are

```math
\tilde p_{t,a}
=
\begin{cases}
p_t^{\mathrm{stay}}, & a=a_{t-1},\\
(1-p_t^{\mathrm{stay}})c_{t,a}, & a\ne a_{t-1}.
\end{cases}
```

With lapse parameter $\lambda$, the returned distribution is

```math
p_{t,a}
=
(1-\lambda)\tilde p_{t,a}
+
\frac{\lambda}{|A_t|}.
```

On the first trial, all action beliefs are identical, so the model returns a
uniform distribution.

## 5. Frozen parameter values

The submitted artifact contains the following final parameters:

| Parameter | Value | Operational role |
|---|---:|---|
| `initial_value` | 50.000000 | Initial mean reward belief |
| `initial_variance` | 400.000000 | Initial uncertainty |
| `process_variance` | 25.000000 | Fixed trial-to-trial uncertainty growth |
| `observation_variance` | 200.000000 | Fixed reward-observation noise |
| `value_sensitivity` | 1.097430 | Value contribution to action utility |
| `uncertainty_sensitivity` | 0.021051 | Relative-uncertainty contribution |
| `stay_intercept` | -0.738172 | Baseline stay-gate intercept |
| `stay_value_weight` | 1.874595 | Effect of previous-action base log-odds |
| `surprise_linear_weight` | 1.489112 | Linear reward-surprise effect |
| `surprise_quadratic_weight` | -1.100678 | Nonlinear reward-surprise effect |
| `run_length_weight` | 1.298749 | Effect of current choice-run length |
| `trial_index_weight` | 0.267720 | Within-trajectory time trend |
| `lapse` | 0.013091 | Uniform random-response mixture |
| `variance_ceiling` | 10000.000000 | Numerical variance cap |

The small fitted uncertainty coefficient indicates limited public-data
support for directed exploration in this specification. Its presence should
not be interpreted as strong evidence that uncertainty drives behavior.

## 6. Model evolution and ablation evidence

### 6.1 What changed from StoVol-Lite v0.1

StoVol-Lite v0.1 used a single four-action softmax combining Kalman value,
relative uncertainty, an accumulating choice trace, reward-stay bias, and a
heuristic adaptive-noise rule. Stay-Switch Kalman v0.2:

- retains Kalman value learning and uncertainty tracking;
- replaces the accumulating choice trace with log run length;
- confines perseveration to a stay-switch gate;
- models the switch target conditionally among non-previous actions;
- replaces simple reward-stay bias with signed and squared standardized
  reward surprise;
- adds a descriptive log trial-count term; and
- removes heuristic online adaptation of process and observation noise.

The architecture therefore changed at the policy level; this is not only a
parameter retuning of the earlier model.

### 6.2 Adaptive-noise ablation

The earlier model attempted to reallocate recent prediction error between
process noise and observation noise during evaluation. A local public-data
ablation compared that specification with an otherwise matched version whose
`noise_adaptation_rate` was set to zero:

| Developmental comparison | Mean NLL |
|---|---:|
| Full StoVol-Lite v0.1 | 0.641966 |
| StoVol-Lite with adaptive noise disabled | 0.638966 |
| Fixed-noise minus adaptive-noise NLL | -0.003000 |

Lower NLL is better. Under this comparison, disabling the tested adaptive-noise
rule slightly improved prediction. The rule was therefore not retained in
v0.2; process and observation variances are instead selected offline and
frozen before evaluation.

This is a negative result about one particular heuristic, not evidence that
human participants never adapt to volatility or that volatility and choice
stochasticity are psychologically absent. The comparison used public data and
should be treated as developmental evidence rather than an independent
confirmatory test.

### 6.3 Predictive comparison

| Public-data diagnostic | StoVol-Lite v0.1 | Stay-Switch Kalman v0.2 |
|---|---:|---:|
| Mean NLL | 0.641966 | 0.569410 |
| Official-convention accuracy | 79.0912% | 79.7015% |
| Geometric mean probability assigned to the observed action | 52.63% | 56.59% |

The approximately 11.3% reduction in public-data NLL is consistent with better
probability calibration, especially for the stay-switch decision. These
numbers are not hidden-test results.

## 7. Fitting and public-data evidence

The recorded development procedure used a deterministic 80/20 split by
anonymized `subject_id` with seed 2026. No subject appeared in both
development partitions. Decision parameters were fitted by bounded maximum
likelihood with a small L2 penalty. A small grid over fixed process and
observation variances was compared by validation NLL. After selecting those
variance values, decision parameters were refitted on all public trajectories
and frozen in `artifacts/fitted_params.json`.

| Recorded evaluation | Subjects | Trials | Mean NLL | Official-convention accuracy |
|---|---:|---:|---:|---:|
| Subject-disjoint training partition | 551 | 256,455 | 0.577997 | 79.3521% |
| Subject-disjoint validation partition | 138 | 63,625 | 0.534969 | 81.0158% |
| Final fit evaluated on all public data | 689 | 320,080 | 0.569410 | 79.7015% |

The validation partition happened to be easier than the training partition,
as indicated by both lower NLL and higher accuracy. In addition, the split was
used during model development and variance selection. Its result is therefore
a development estimate, not a fully independent final estimate.

For the final public fit:

| Trial category | Trials | Mean NLL |
|---|---:|---:|
| First trial | 2,678 | 1.386294 |
| Stay trial | 239,937 | 0.180043 |
| Switch trial | 77,465 | 1.747178 |

Most predictive performance comes from stay trials. Switch-target prediction
remains the primary empirical weakness and the most promising focus for a
future challenger model.

For reference, uniform prediction over four actions has mean NLL
$-\log(0.25)=1.386294$. The final mean NLL of 0.569410 corresponds to a
geometric mean probability of approximately
$\exp(-0.569410)=0.5659$ assigned to the observed action.

## 8. Why the model is cognitive

The model contains explicit, inspectable latent beliefs and a fixed
trial-by-trial update rule. Each fitted parameter has an operational mapping
to value learning, uncertainty, reward surprise, persistence, time trend, or
lapse. Every prediction can be reconstructed from the current trajectory's
previously revealed observations.

The cognitive interpretation remains conditional:

- similar predictions may be produced by simpler choice kernels or
  win-stay-lose-shift rules;
- fitted parameters may trade off against one another;
- predictive superiority alone does not establish psychological uniqueness;
  and
- the trial-count effect is descriptive rather than a direct measure of a
  named mental process.

## 9. Alternative explanations and discriminative tests

| Alternative explanation | Useful discriminative test |
|---|---|
| Choice persistence alone explains the result. | Compare against a reward-free run-length or choice-kernel model on subject-held-out data. |
| Win-stay-lose-shift is sufficient. | Compare the continuous surprise terms with a binary previous-outcome rule. |
| Standard Q-learning is sufficient. | Replace Kalman beliefs with fixed-learning-rate Q values while retaining the same two-stage policy. |
| The two-stage policy is unnecessary. | Compare against a single multi-action softmax using identical Kalman beliefs and matched parameter complexity. |
| Uncertainty does not contribute. | Set `uncertainty_sensitivity` to zero and compare held-out NLL. |
| Stable participant subtypes explain persistence. | Evaluate a prespecified hierarchical or mixture model on unseen participants. |

Mechanism-specific probes should also verify that:

- changing run length alters the stay gate but not the conditional ranking of
  switch targets;
- changing the previous reward surprise alters stay probability while holding
  the current value state fixed; and
- changing uncertainty between two non-previous actions affects their
  conditional switch probabilities independently of perseveration.

## 10. Failure conditions and limitations

- Hidden tasks may use different reward scales, numbers of actions,
  volatility, or participant strategies.
- A single global parameter set cannot represent all stable individual
  differences in exploration and perseveration.
- The fixed random-walk Kalman dynamics may be misspecified when values
  mean-revert, jump abruptly, or follow structured correlations.
- Strong public-data run-length effects may not transfer to a task with
  weaker choice inertia.
- First-trial prediction is uniform because no participant-specific or
  task-specific pre-choice evidence is available.
- Switch-trial NLL is substantially worse than stay-trial NLL.
- The small uncertainty coefficient limits claims about directed exploration.
- The adaptive-noise ablation rejects only the tested heuristic; it does not
  settle whether a better volatility-learning model would generalize.
- The final all-public score is evaluated on data used for final parameter
  fitting and is not an unbiased generalization estimate.

## 11. Reproducibility and leakage controls

- Agent entry point: `agent.py`.
- Runtime configuration: `config.yaml`.
- Frozen parameters and aggregate diagnostics:
  `artifacts/fitted_params.json`.
- Offline fitting and subject-disjoint model selection: `fit_public.py`.
- Sequential public-data evaluation: `evaluate_public.py`.
- Interface, probability, dynamic-import, missing-reward, and reset tests:
  `tests/test_agent.py`.
- Prediction is produced before the current action and reward are supplied to
  `update()`.
- Fitting uses public trajectories only and groups the development split by
  `subject_id`.
- No optimizer, parameter search, or model-weight update occurs inside
  `Agent` during evaluation.
- `reset()` clears all mutable trajectory-local state.
- The action set is read from `context.available_actions`; it is not hard-coded
  to four actions.
- No raw public trajectories are included in the frozen parameter artifact.
- No hidden evaluation data, source mappings, credentials, or identifying
  participant information are used.

Reproduction commands from the repository root are:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python evaluate_public.py
```

Offline refitting additionally requires:

```powershell
python -m pip install -r requirements-fit.txt
python fit_public.py `
  --process-variance-grid 10,25,50 `
  --observation-variance-grid 50,100,200
```

## 12. Confidentiality and reporting

This card reports public-data development evidence only. It contains no hidden
benchmark result, raw trajectory, source-identifying mapping, credential,
demographic variable, or original participant identifier.

When citing the numerical results, use language such as:

> Local public-data diagnostic under the recorded fitting procedure; not an
> official hidden-evaluation score.
