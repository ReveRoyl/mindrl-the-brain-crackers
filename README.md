# The Brain Crackers — History-Augmented Stay-Switch Kalman

A choice-only cognitive model with fixed-noise Kalman beliefs and causal
within-trajectory reward/choice history. The stay/switch factorization is a
convenient predictive representation, not proof of two psychological stages.

## Run the final model

Use Python 3.10–3.12. From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python evaluate_public.py
```

Place the authorized public input at `data/public_train.jsonl`.
The agent itself uses only the standard library; PyYAML supports configuration.
No network, external weights, API or GPU is needed for prediction.

Expected full-public fit diagnostic:

```text
Trajectories: 2678
Trials: 320080
Mean NLL: 0.560104
Official accuracy: 79.8883%
```

This is an in-sample diagnostic, not a hidden-test score.
The selected candidate's five-fold subject-grouped development result is
NLL **0.560396**, accuracy **79.8775%**, versus a same-fold refitted baseline
of **0.569687**, **79.6825%**. The complete candidate comparison and selection
caveats are in the [interpretation card](interpretation_card.md).

To run the preserved reference with unchanged checkpoint parameters:

```powershell
python evaluate_public.py --config config_baseline.yaml
```

Expected reference: NLL 0.569410, accuracy 79.7015%.

## Repository guide

| File | Purpose |
|---|---|
| [agent.py](agent.py) | Standalone official Agent entry point; only trajectory state updates online. |
| [config.yaml](config.yaml) | Final configuration and parameter path. |
| [final parameters](artifacts/fitted_params_final.json) | Thirteen offline-fitted decision coefficients and fixed belief settings. |
| [CV record](artifacts/final_development_cv.json) | Every candidate/fold, optimizer attempts, paired intervals and selection. |
| [sequential score](artifacts/final_public_evaluation.json) | Runtime score, loss decomposition and content hashes. |
| [experiments](experiments/) | Offline fitting, numerical checks and audit arithmetic verification. |
| [data checks](analysis/final_data_checks.ipynb) | Executed notebook; aggregate integrity checks, no raw-data output. |
| [tests](tests/) | Interface, reset, causal order, exact ties, numerical parity and math syntax. |
| [submission metadata](submission.yaml) | Team, visibility, runtime and artifact declarations. |
| [post-freeze changes](post_freeze_changes.md) | Frozen-target/final-package distinction and audit handling. |
| [labbook](labbook/README.md) | Protocol, negative results and audit reply drafts. |
| [submission checklist](FINAL_SUBMISSION_CHECKLIST.md) | Team review, commit pinning and formal final intake. |

## Reproduce development

The runtime-only environment skips six offline numerical tests explicitly.
Install the fitting dependencies to run the full suite and reproduce CV:

```powershell
python -m pip install -r requirements-fit.txt
python -m unittest discover -s tests -v
python -m experiments.final_candidates --workers 4 --refit --output artifacts/reproduced_cv.json --parameter-output artifacts/refitted_candidate.json
python -m experiments.validate_final
```

This fits six candidates plus a fixed equal-weight mixture on the same five
subject folds. Parameters are initialized independently of the all-public
artifact. New outputs do not replace deployed or checkpoint parameters.
The saved reference environment is Python 3.11.5, NumPy 2.2.6 and SciPy 1.15.3;
other supported versions may have small numerical differences.
Notebook execution additionally needs Jupyter/nbconvert, not runtime evaluation.

The original `fit_public.py` is retained for the nine-coefficient reference;
use `--config config_baseline.yaml` and an explicit **new** `--output` if
rerunning it. Its default output points to the preserved checkpoint artifact,
so do not run it casually without `--output`.

## Official interface and submission

```python
agent = Agent(config)
agent.reset(context)
prediction = agent.predict(history)
agent.update(action, reward, info)
```

Probabilities cover the available actions and sum to one. The model does not
use subject identity, response time, reward-schedule sidecars or future history.
All state clears on reset.

Pushing code is not formal submission. After team approval, pin the exact
40-character published commit through the official rolling-submission form.
Audit reply drafts are local and have not been posted automatically.
