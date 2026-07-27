# The Brain Crackers — Stay–Switch Kalman Agent

This repository contains a runnable cognitive reinforcement-learning submission
for the MindRL Challenge. The model first predicts whether the participant
repeats the preceding action and then, conditional on switching, compares the
remaining actions using Kalman value and uncertainty estimates.

## Repository contents

| Path | Purpose |
|---|---|
| `agent.py` | Official `Agent` implementation used during evaluation. |
| `config.yaml` | Runtime settings and local frozen-parameter path. |
| `artifacts/fitted_params.json` | Frozen global parameters fitted offline on public data. |
| `fit_public.py` | Subject-disjoint fitting and model-selection pipeline. |
| `evaluate_public.py` | Sequential public-data evaluation with official NLL and accuracy conventions. |
| `tests/test_agent.py` | Dynamic-loader, probability, update, and reset tests. |
| `submission.yaml` | Team, method, runtime, and artifact metadata. |
| `interpretation_card.md` | Scientific claims, mechanisms, evidence, and limitations. |

## Runtime installation

Create and activate a Python 3.10–3.12 environment, then run:

```powershell
python -m pip install -r requirements.txt
```

The submitted agent itself uses only the Python standard library. PyYAML is
used by the local evaluation scripts and the official validation workflow.

## Run the tests

From the repository root:

```powershell
python -m unittest discover -s tests -v
```

The loader test deliberately imports `agent.py` without first adding the
module to `sys.modules`, matching the compatibility condition that broke the
earlier dataclass implementation.

## Evaluate the frozen model

Place the public data at `data/public_train.jsonl`, then run:

```powershell
python evaluate_public.py
```

Expected results for the included frozen artifact are approximately:

```text
Trajectories: 2678
Trials: 320080
Mean NLL: 0.569410
Official accuracy: 79.7015%
```

The result on all public data is a fit diagnostic and is not a hidden-test
score.

## Refit from public data

Install the offline fitting dependencies:

```powershell
python -m pip install -r requirements-fit.txt
```

Run the recorded subject-disjoint development procedure:

```powershell
python fit_public.py
```

To repeat the small Kalman-variance grid used for the included artifact:

```powershell
python fit_public.py `
  --process-variance-grid 10,25,50 `
  --observation-variance-grid 50,100,200
```

The fitting script writes only global parameter estimates and aggregate
diagnostics to `artifacts/fitted_params.json`. It does not copy raw
trajectories into the submission artifact.

## Official interface

The evaluator uses:

```python
agent = Agent(config)
agent.reset(context)
prediction = agent.predict(history)
agent.update(action, reward, info)
```

`prediction["action_probs"]` contains one finite, non-negative probability for
every available action and sums to one.
