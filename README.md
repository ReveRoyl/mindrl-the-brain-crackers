# MindRL Challenge Submission Repository

This is your team's private submission repository for the MindRL Hub Modeling Challenge. Use it to develop your model, document your method, and pin the exact commit organizers should evaluate.

Public dataset homepage: https://huggingface.co/datasets/mindrl-hub/mindrl-challenge-public

Use `public_train.jsonl` for your own training, debugging, and public sanity checks. Organizers may also run public-reference checks, but official ranking uses hidden evaluation splits that are not released to teams.

## What To Edit

1. Replace or extend `agent.py` with your model. Keep a top-level `Agent` class.
2. Update `config.yaml` with the settings your agent needs.
3. Fill `submission.yaml` with your real team metadata, method description, runtime profile, repo URL, and final commit hash.
4. Complete `interpretation_card.md` with the claims, mechanisms, limitations, and evidence for your method.
5. Keep `requirements.txt` minimal but complete: list packages your code imports.
6. Keep large checkpoints outside git and reference them from `submission.yaml` or `config.yaml`.

Do not commit API keys, private data, hidden benchmark files, raw source-identifying information, original participant ids, original block ids, demographics, or source mappings.

## Agent API

Evaluation is always one-step-ahead prediction within the current trajectory:

```text
P(action_t | context, history_1:t-1)
```

The evaluator calls your agent sequentially:

```python
agent.reset(context)
history = []
for trial in trajectory:
    prediction = agent.predict(history)      # predict before seeing this trial outcome
    agent.update(trial.action, trial.reward, trial.info)
    history.append(trial)
```

Your `Agent` must expose:

```python
class Agent:
    def __init__(self, config=None): ...
    def reset(self, context): ...
    def predict(self, history): ...
    def update(self, action, reward, info=None): ...
```

`predict(history)` must return probabilities over exactly `context.available_actions`:

```python
{"action_probs": {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}}
```

Read the action set from `context.available_actions`; do not assume every evaluation task has exactly four actions.

## Optional Response-Time Evaluation

Response-time (RT) modeling is optional and is reported on a separate leaderboard. It does not change the primary choice leaderboard, and choice-only submissions remain fully valid.

To opt in, add this root-level field to `submission.yaml`:

```yaml
optional_targets:
  - response_time
```

Then return a positive, finite RT prediction in milliseconds on every call to `predict()`:

```python
{
    "action_probs": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25},
    "rt_ms": 715.0,
}
```

An agent may use RTs from completed prior trials as online context. The current trial's observed RT is not available until after that trial's prediction. See the [full optional RT evaluation specification](https://github.com/mindrl-challenge/mindrl-challenge-core/blob/main/docs/optional_rt_evaluation.md) for validation, scoring, and temporal-boundary details.
## Public Data

Download the public trajectory file:

```bash
huggingface-cli download mindrl-hub/mindrl-challenge-public public_train.jsonl --local-dir ./hf_cache/public
```

Each line in `public_train.jsonl` is one trajectory with:

- `context`: task id, task family, anonymized subject/trajectory ids, available actions, optional features, and metadata.
- `trials`: chronological observed human choices, rewards, and optional response-time information.

The public dataset is source-safe. Do not try to identify the original study, raw data source, collection site, institution, participant pool, or hidden split origin.

## Hidden Evaluation Conditions

Hidden evaluation may include several generalization conditions. The evaluator may expose the condition as:

```python
metadata = getattr(context, "metadata", {}) or {}
condition = metadata.get("generalization_condition", "unknown")
```

Expected values:

| Value | What it means for your agent |
| --- | --- |
| `public_train` | Released public training trajectory. |
| `continuation` | Hidden continuation-style segment. The first hidden prediction still starts with `history=[]`; no public prefix history is automatically provided. |
| `unseen_participant` | Hidden public-task trajectory from a participant not seen in public training. Do not rely on subject-specific lookup tables. |
| `heldout_block` | Hidden public-task episode/block. The evaluator still calls `reset(context)` for the current trajectory. |
| `heldout_task` | Hidden trajectory from a different task family or task structure. Read `available_actions`, `num_options`, `task_family`, `task_description`, and `option_features`; do not hard-code public-task assumptions. |
| `unknown` | Fallback; handle missing or unfamiliar labels gracefully. |

You may condition priors, adaptation speed, prompts, model heads, or fallback behavior on this field. You must still adapt only online within the current trajectory after outcomes are revealed.

A robust `reset` pattern:

```python
def reset(self, context):
    self.context = context
    self.actions = list(context.available_actions)
    metadata = getattr(context, "metadata", {}) or {}
    self.condition = metadata.get("generalization_condition", "unknown")
```

## Local Validation And Public Evaluation

The validation/evaluation scripts live in the core repository, not in this team repo. From a local clone of `mindrl-challenge-core`, validate your current committed repo state with toy data:

```bash
python scripts/validate_submission.py \
  --repo-url https://github.com/mindrl-challenge/submission-the-brain-crackers.git \
  --commit-hash <commit_sha> \
  --agent-path agent.py \
  --config-path config.yaml \
  --requirements-path requirements.txt \
  --data examples/toy_data/toy_bandit_trajectories.jsonl \
  --output validation_report.json
```

To run a capped public-data evaluation from the core repository, use the same clone-and-wrapper path organizers use:

```bash
python scripts/evaluate_submission.py \
  --repo-url https://github.com/mindrl-challenge/submission-the-brain-crackers.git \
  --commit-hash <commit_sha> \
  --agent-path agent.py \
  --config-path config.yaml \
  --requirements-path requirements.txt \
  --eval-config configs/evaluation_config.public_hf_splits.example.yaml \
  --output public_scores.json \
  --max-trajectories-per-split 20
```

Remove `--max-trajectories-per-split 20` when you want to run the full public split. Public scores are useful for debugging and sanity checks. They are not the official ranking.

If you are on Windows and `python` points to the Microsoft Store shim or another wrong interpreter, use `py -3` in place of `python`.

## Runtime Profile

Every submission must include a `runtime_profile` block in `submission.yaml` so organizers can route evaluation before cloning or running your repository.

```yaml
runtime_profile:
  execution_type: local_python        # local_python | gpu_model | external_api | hybrid
  model_family: symbolic_cognitive    # symbolic_cognitive | neural | llm_agent | auditing | hybrid | other
  requires_gpu: false
  gpu_type: null
  requires_external_api: false
  required_secrets: []                # e.g. ["OPENAI_API_KEY"]; list names only, never values
  estimated_eval_cost: none
  expected_runtime_minutes: null
  notes: "Describe any special evaluation handling here."
```

External API or GPU submissions must still expose a callable `Agent` class. Do not commit secrets.


## How Rolling Submission Works

Official submissions are self-service through the **Rolling submission** issue form in mindrl-challenge-core. Your model code stays in this repository; the core repository records only a pinned registry row.

1. Push your current work to this repository.
2. Run git rev-parse HEAD in this repository and copy the full 40-character commit SHA.
3. Put the same SHA in submission.yaml under submission.commit_hash and make sure submission.repo_url is this repository URL.
4. Open a **Rolling submission** issue in the core repository: https://github.com/mindrl-challenge/mindrl-challenge-core/issues/new/choose
5. Fill in the registry fields: team id, display name, visibility, repo URL, commit hash, file paths, method family, and short description.
6. The intake workflow opens or updates a registry PR in mindrl-challenge-core that changes only submissions_registry.yaml.
7. CI runs toy validation on the pinned commit. Organizers review and merge the registry PR before the submission enters the evaluation queue.

The registry entry should point to files inside this repository, usually:

``yaml
repo_url: https://github.com/mindrl-challenge/submission-the-brain-crackers.git
commit_hash: <full_commit_sha>
agent_path: agent.py
config_path: config.yaml
requirements_path: requirements.txt
submission_metadata_path: submission.yaml
interpretation_card_path: interpretation_card.md
``

If you change code after submitting a commit hash, run git rev-parse HEAD again and submit a new rolling issue or update the still-open issue. Organizers evaluate the pinned commit, not whichever branch happens to be latest.


## Final Submission Checklist

Before the final submission deadline:

- Commit and push your final code to this repository.
- Run `git rev-parse HEAD` and copy that exact commit SHA into `submission.yaml`.
- Make sure `submission.yaml` points to the right `agent_path`, `config_path`, `requirements_path`, and `interpretation_card_path`.
- Make sure `runtime_profile` accurately says whether your method needs GPU, external APIs, secrets, large artifacts, or special evaluation handling.
- Confirm validation passes on toy data from the core repository.
- Open a **Rolling submission** issue in `mindrl-challenge-core` and confirm that the generated registry PR passes validation CI.

## Required Files

| File | Purpose |
| --- | --- |
| `agent.py` | Your `Agent` implementation. |
| `config.yaml` | Runtime and model settings. |
| `submission.yaml` | Team metadata, paths, repo URL, commit hash, and runtime profile. |
| `requirements.txt` | Python dependencies imported by your code. |
| `interpretation_card.md` | Scientific interpretation, claims, limitations, and evidence. |
| `LICENSE_OR_POLICY_NOTICE.md` | Challenge evaluation and IP notice. |
| `.gitignore` | Keeps scores, logs, caches, secrets, and large local artifacts out of git. |

Optional small artifacts may live under `artifacts/`. Large checkpoints should be hosted externally on an organizer-approved service and referenced in metadata.