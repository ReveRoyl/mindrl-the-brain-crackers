# MindRL Challenge Submission Template
<!-- MINDRL_PARTICIPANT_QUICKSTART_START -->
## Participant Quick Start: Public Data and Hidden Evaluation API

Public dataset homepage: https://huggingface.co/datasets/mindrl-hub/mindrl-challenge-public

Use `public_train.jsonl` for your own training, debugging, and public sanity checks. Organizers may also run public-reference checks, but official ranking uses hidden evaluation splits that are not released to teams.

Download the public trajectory file:

```bash
huggingface-cli download mindrl-hub/mindrl-challenge-public public_train.jsonl --local-dir ./hf_cache/public
```

Run a local public evaluation from the core repository:

```bash
python -m mindrl.evaluator.runner \
  --agent path/to/your/agent.py \
  --config path/to/your/config.yaml \
  --data ./hf_cache/public/public_train.jsonl \
  --output public_scores.json
```

### Evaluation API

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

`predict(history)` must return probabilities over exactly `context.available_actions`, for example:

```python
{"action_probs": {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}}
```

### Hidden evaluation conditions

Hidden evaluation may include several generalization conditions. The evaluator may expose the condition as:

```python
condition = context.metadata.get("generalization_condition", "unknown")
```

Expected values:

| Value | What it means for your agent |
| --- | --- |
| `public_train` | Released public training trajectory. |
| `continuation` | Hidden continuation-style segment. The first hidden prediction still starts with `history=[]`; no public prefix history is automatically provided. |
| `unseen_participant` | Hidden public-task trajectory from a participant not seen in public training. Do not rely on subject-specific lookup tables. |
| `heldout_block` | Hidden public-task episode/block. The evaluator still calls `reset(context)` for the current trajectory. |
| `heldout_task` | Hidden trajectory from a different task family or task structure. Read `available_actions`, `num_options`, `task_family`, `task_description`, and `option_features`; do not hard-code four actions. |
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

Do not use hidden trajectories, current/future trial outcomes, raw-data source reconstruction, source-identifying information, original participant ids, original block ids, demographics, or source mappings.
<!-- MINDRL_PARTICIPANT_QUICKSTART_END -->
This repository is the **official participant submission template** for the MindRL Challenge. In its current form it is a **valid minimal submission** that implements a **Win-Stay-Lose-Shift (WSLS)** cognitive baseline, suitable for testing the official `validate_submission.py` script from the **core evaluator** repository (this repo does not ship that script).

## 1. What this repository is

A **lightweight artifact** that defines what each team turns in: a callable agent module, runtime configuration, machine-readable submission metadata, dependencies, and an interpretation card. It does **not** include the benchmark evaluator, dataset pipeline, training code, leaderboard infrastructure, or hidden evaluation data.

## 2. WSLS baseline in this template

- **`agent.py`** defines `Agent` with the official API: `__init__`, `reset`, `predict`, `update`.
- **`predict(history)`** returns `{"action_probs": {action: probability, ...}}` with valid probabilities that sum to 1 (after an epsilon floor on every action).
- With **no history**, the policy is **uniform** over `context["available_actions"]` (or `[1, 2, 3, 4]` if none are provided).
- With **history**, the last trial’s reward is compared to `model.win_threshold`; above threshold the model favors **staying** on the last action (`stay_probability`); below threshold it favors **shifting** away (complementary mass split across other actions). See `interpretation_card.md` for the scientific framing.

## 3. Required files

| File | Purpose |
|------|---------|
| `agent.py` | Your `Agent` class (evaluator imports this). |
| `config.yaml` | Runtime and model-related settings. |
| `submission.yaml` | Team and submission metadata for organizers. |
| `requirements.txt` | Python dependencies (keep minimal). |
| `interpretation_card.md` | Structured scientific interpretation (community-visible by default). |
| `LICENSE_OR_POLICY_NOTICE.md` | Restricted evaluation and IP notice. |
| `.gitignore` | Keeps logs, caches, and local artifacts out of git. |

Optional: `artifacts/` for small local files (see [Artifact / checkpoint guidance](#8-artifact--checkpoint-guidance)).

## 4. Agent API

The official API is:

```python
class Agent:
    def __init__(self, config=None)
    def reset(self, context)
    def predict(self, history)
    def update(self, action, reward, info=None)
```

**Semantics**

- **`context`** — Provided **once** at the start of a trajectory (e.g. action space, `available_actions`, task identifiers). Use it to set up anything that is fixed for that episode or block.
- **`history`** — Contains **only past trials** (not the current decision step's outcome). `predict` bases its distribution on this history when present; `update` may cache outcomes but the WSLS policy is driven primarily from **history** for evaluator compatibility.
- **`predict`** — Must return **action probabilities** over valid actions for this step.
- **`update`** — Called **after** the true action and reward (and optional `info`) are revealed.

**Evaluation rule**

Parameters are fixed before evaluation; online state may update within a trajectory when `evaluation.allow_online_state_updates` is honored by the evaluator.

## 5. Replacing the baseline with your own model

1. **Replace `agent.py`** — Keep a top-level `Agent` class with the same four methods and the same `predict` return shape (`action_probs` dict). You may add modules alongside `agent.py` if the evaluator allows; keep imports resolvable via `requirements.txt`.
2. **Edit `config.yaml`** — Add or rename keys under `model` as your loader expects; keep `runtime` / `evaluation` as required by the challenge.
3. **Fill `submission.yaml`** — Replace placeholder `repo_url`, `commit_hash`, and team fields with real values before an actual submission.
4. **Complete `interpretation_card.md`** — Replace or extend the WSLS example sections with claims and evidence for your method (the current card documents the template WSLS only).

## 6. Runtime profile

Every submission must include a `runtime_profile` block in `submission.yaml`. The same summary is copied into the official registry entry so organizers can route evaluations before cloning the repository.

```yaml
runtime_profile:
  execution_type: local_python        # local_python | gpu_model | external_api | hybrid
  model_family: symbolic_cognitive    # symbolic_cognitive | neural | llm_agent | auditing | hybrid | other
  requires_gpu: false
  gpu_type: null
  requires_external_api: false
  required_secrets: []                # e.g. ["OPENAI_API_KEY"]; leave empty when none
  estimated_eval_cost: none           # none | low | medium | high, or a short estimate
  expected_runtime_minutes: null
  notes: "Describe any special evaluation handling here."
```

The runtime profile does **not** change the required `Agent` API. Even LLM/API/GPU submissions must expose a callable `Agent` class. Do not commit API keys or secrets. If your approach needs external APIs, list secret names only and describe expected cost or rate limits.

## 7. Organizer validation (`validate_submission.py`)

Validation is performed using the **core evaluator repository**, not by running a script inside this submission repo. From the **core** repo root, a typical invocation looks like:

```bash
python scripts/validate_submission.py \
  --repo-url <this-repo-url> \
  --commit-hash <commit-hash> \
  --agent-path agent.py \
  --config-path config.yaml \
  --requirements-path requirements.txt \
  --data examples/toy_data/toy_bandit_trajectories.jsonl \
  --output validation_report.json
```

Adjust `--data` to a **toy or public** path provided by the organizers. Do not commit private evaluation sets.

## 8. How to run other local checks

If the core package exposes a runner module, you can also smoke-test the agent there (exact CLI depends on the released evaluator). Replace data paths with allowed public or toy splits only.

## 9. Artifact / checkpoint guidance

- **Small** auxiliary files may live under `artifacts/` (see `artifacts/README.md`).
- **Large** checkpoints belong on Hugging Face (or another organizer-approved host): record the URI in `submission.yaml` / `config.yaml` and keep this git repository small.
- Never commit API keys, SSH private keys, or raw identifiable participant data.

## 10. Visibility and confidentiality

Default visibility in the template `submission.yaml` is **`internal`**: shared for evaluation, review, adversarial analysis, mentorship, and challenge-internal discussion. Read `LICENSE_OR_POLICY_NOTICE.md` for the evaluation-only IP framing. If your team later opts into public release, update `submission.yaml` and your own licensing accordingly.

## 11. Submission checklist

- [ ] `Agent` API matches the specification and loads from `agent.py`.
- [ ] `config.yaml` reflects your runtime (seed, device, checkpoint references).
- [ ] `submission.yaml` has a complete `runtime_profile` matching your evaluation needs.
- [ ] `submission.yaml` has real `repo_url` / `commit_hash` (and team fields) for real submissions.
- [ ] `interpretation_card.md` is complete and non-sensitive.
- [ ] `requirements.txt` lists all **imported** dependencies; avoid unused heavy packages.
- [ ] No secrets, private data, or hidden benchmark files in the repo.
- [ ] Validation run completed using the official core script and allowed data path.
- [ ] `scores.json` (or similar) is gitignored and not submitted unless instructions say so.
