"""Pre-specified public development CV; never overwrites the frozen artifact.

Run from the repository root: python -m experiments.final_candidates
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import yaml
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from fit_public import PARAMETER_BOUNDS, PARAMETER_NAMES, config_for_feature_extraction, load_trajectories, parameters_from_config
from experiments.history_agent import EXTRA_NAMES, HistoryAgent

NAMES = PARAMETER_NAMES + EXTRA_NAMES
UTILITY = [0, 1, 11, 12]
GATE = [2, 3, 4, 5, 6, 7, 9, 10]
REGULARIZED = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
SEED = 20260827
CANDIDATES = {
    "baseline": {"extra": [], "r": 200.0, "l2": 1e-5},
    "history_gate": {"extra": [9, 10], "r": 200.0, "l2": 1e-5},
    "history_target": {"extra": [11, 12], "r": 200.0, "l2": 1e-5},
    "history_full": {"extra": [9, 10, 11, 12], "r": 200.0, "l2": 1e-5},
    "history_regularized": {"extra": [9, 10, 11, 12], "r": 200.0, "l2": 1e-3},
    "history_r100": {"extra": [9, 10, 11, 12], "r": 100.0, "l2": 1e-5},
}


@dataclass
class Batch:
    utility: np.ndarray
    gate_static: np.ndarray
    last: np.ndarray
    observed: np.ndarray
    subject: np.ndarray
    first_subject: np.ndarray

    @property
    def n_actions(self):
        return self.utility.shape[1]

    def subset(self, subjects):
        mask = np.isin(self.subject, subjects)
        first_mask = np.isin(self.first_subject, subjects)
        return Batch(self.utility[mask], self.gate_static[mask], self.last[mask], self.observed[mask], self.subject[mask], self.first_subject[first_mask])


def extract(trajectories, config, subject_ids):
    """No global fitted file, outcome-derived current features, or ID predictors."""
    if config["model"].get("fitted_params_path"):
        raise ValueError("Feature extraction must disable full-data artifact loading")
    agent = HistoryAgent(config)
    utility, gate, last, observed, subject, first = [], [], [], [], [], []
    n_actions = None
    for trajectory in trajectories:
        agent.reset(trajectory["context"])
        sid = subject_ids[trajectory["context"]["subject_id"]]
        actions = agent.actions
        if n_actions is None:
            n_actions = len(actions)
        if len(actions) != n_actions or n_actions < 2:
            raise ValueError("Offline fitting requires a constant action count >= 2")
        for trial in trajectory["trials"]:
            if trial["action"] not in actions:
                raise ValueError("Unavailable observed action")
            if not agent.trial_count:
                first.append(sid)
            else:
                values, uncertainties = agent._standardized_features()
                persistence, trace, frequency, recency = agent.history_features()
                utility.append([[values[a], uncertainties[a], frequency[a], recency[a]] for a in actions])
                gate.append([1.0, agent.last_surprise, agent.last_surprise**2, math.log1p(agent.run_length), math.log1p(agent.trial_count), persistence, trace])
                last.append(actions.index(agent.last_action))
                observed.append(actions.index(trial["action"]))
                subject.append(sid)
            agent.update(trial["action"], trial.get("reward"), trial.get("info"))
    batch = Batch(np.asarray(utility), np.asarray(gate), np.asarray(last, dtype=int), np.asarray(observed, dtype=int), np.asarray(subject, dtype=int), np.asarray(first, dtype=int))
    if not np.isfinite(batch.utility).all() or not np.isfinite(batch.gate_static).all():
        raise ValueError("Nonfinite extracted feature")
    return batch


def terms(theta, batch):
    rows = np.arange(len(batch.last))
    utilities = np.einsum("nad,d->na", batch.utility, theta[UTILITY])
    switched = utilities.copy()
    switched[rows, batch.last] = -np.inf
    denominator = logsumexp(switched, axis=1)
    conditional = np.exp(switched - denominator[:, None])
    raw_logodds = utilities[rows, batch.last] - denominator
    lower, upper = math.log(1e-9 / (1.0 - 1e-9)), math.log((1.0 - 1e-9) / 1e-9)
    logodds = np.clip(raw_logodds, lower, upper)
    gate_features = np.column_stack((batch.gate_static[:, 0], logodds, batch.gate_static[:, 1:]))
    stay_probability = expit(np.einsum("nd,d->n", gate_features, theta[GATE]))
    raw = (1.0 - stay_probability)[:, None] * conditional
    raw[rows, batch.last] = stay_probability
    probabilities = (1.0 - theta[8]) * raw + theta[8] / batch.n_actions
    return probabilities, raw, conditional, stay_probability, gate_features, (raw_logodds > lower) & (raw_logodds < upper)


def objective(theta, batch, penalty):
    probabilities, raw, conditional, s, gate_features, unclipped = terms(theta, batch)
    rows = np.arange(len(batch.last))
    stay = batch.last == batch.observed
    c = conditional[rows, batch.observed]
    p = np.clip(probabilities[rows, batch.observed], 1e-12, 1.0)
    dloss_raw = -(1.0 - theta[8]) / p
    d_raw_gate = np.where(stay, s * (1.0 - s), -c * s * (1.0 - s))
    dloss_gate = dloss_raw * d_raw_gate
    gradient = np.zeros(len(NAMES))
    gradient[GATE] = np.einsum("n,nd->d", dloss_gate, gate_features) / len(rows)
    expected = np.einsum("na,nad->nd", conditional, batch.utility)
    last_x = batch.utility[rows, batch.last]
    observed_x = batch.utility[rows, batch.observed]
    through_gate = (dloss_gate * theta[3] * unclipped)[:, None] * (last_x - expected)
    through_target = (dloss_raw * (~stay) * (1.0 - s) * c)[:, None] * (observed_x - expected)
    gradient[UTILITY] = np.mean(through_gate + through_target, axis=0)
    gradient[8] = np.mean((raw[rows, batch.observed] - 1.0 / batch.n_actions) / p)
    loss = -float(np.mean(np.log(p))) + penalty * float(np.sum(theta[REGULARIZED]**2))
    gradient[REGULARIZED] += 2.0 * penalty * theta[REGULARIZED]
    return loss, gradient


def fit(batch, config, spec, maxiter):
    start = np.r_[parameters_from_config(config), np.zeros(4)]
    alternative = start.copy()
    alternative[:9] = [1.0, 0.0, 0.0, 1.0, 0.5, -0.5, 0.5, 0.0, 0.05]
    bounds = list(PARAMETER_BOUNDS) + [(-5.0, 5.0) if i in spec["extra"] else (0.0, 0.0) for i in range(9, 13)]
    attempts = []
    solutions = []
    for initial in (start, alternative):
        began = time.monotonic()
        result = minimize(objective, initial, args=(batch, spec["l2"]), method="L-BFGS-B", jac=True, bounds=bounds,
                          options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-7, "maxls": 30})
        attempts.append({"success": bool(result.success), "message": str(result.message), "iterations": int(result.nit), "objective": float(result.fun), "seconds": time.monotonic() - began})
        solutions.append(result)
    successful = [x for x in solutions if x.success and np.isfinite(x.fun)]
    if not successful:
        raise RuntimeError(f"Neither start converged: {attempts}")
    best = min(successful, key=lambda x: x.fun)
    return np.asarray(best.x), attempts


def observation_stats(probabilities, batch):
    rows = np.arange(len(batch.last))
    loss = -np.log(np.clip(probabilities[rows, batch.observed], 1e-12, 1.0))
    winners = probabilities == probabilities.max(axis=1, keepdims=True)
    credit = winners[rows, batch.observed] / winners.sum(axis=1)
    return loss, credit


def metrics(probabilities, batch):
    loss, credit = observation_stats(probabilities, batch)
    stay = batch.last == batch.observed
    rows = np.arange(len(batch.last))
    switch_mass = np.clip(1.0 - probabilities[rows, batch.last], 1e-12, 1.0)
    first = len(batch.first_subject)
    count = len(loss) + first
    gate_loss = -np.log(switch_mass[~stay])
    return {"nll": float((loss.sum() + first * math.log(batch.n_actions)) / count),
            "accuracy": float((credit.sum() + first / batch.n_actions) / count),
            "trials": count, "first_trials": first, "stay_trials": int(stay.sum()), "switch_trials": int((~stay).sum()),
            "stay_nll": float(loss[stay].mean()), "switch_nll": float(loss[~stay].mean()),
            "switch_gate_nll": float(gate_loss.mean()), "switch_target_nll": float((loss[~stay] - gate_loss).mean())}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def paired_bootstrap(candidate, baseline, batch, n_subjects):
    loss, credit = observation_stats(candidate, batch)
    base_loss, base_credit = observation_stats(baseline, batch)
    sums = np.stack([np.bincount(batch.subject, weights=loss - base_loss, minlength=n_subjects), np.bincount(batch.subject, weights=credit - base_credit, minlength=n_subjects)], axis=1)
    counts = np.bincount(batch.subject, minlength=n_subjects) + np.bincount(batch.first_subject, minlength=n_subjects)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n_subjects, size=(2000, n_subjects))
    differences = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    return {name: {"delta": float(sums[:, i].sum() / counts.sum()), "percentile_95": np.quantile(differences[:, i], [0.025, 0.975]).tolist()} for i, name in enumerate(("nll", "accuracy"))}


def run_fold(index, valid_ids, subjects, batches, config, max_iterations):
    """Independent fold worker; only its training partition enters optimization."""
    train_ids = np.setdiff1d(np.arange(len(subjects)), valid_ids)
    assert not set(train_ids) & set(valid_ids)
    record = {"fold": index, "training_subjects": len(train_ids), "validation_subjects": len(valid_ids),
              "validation_membership_sha256": hashlib.sha256("\n".join(str(subjects[i]) for i in sorted(valid_ids)).encode()).hexdigest(), "candidates": {}}
    fold_batches = {r: (b.subset(train_ids), b.subset(valid_ids)) for r, b in batches.items()}
    predictions = {}
    for name, spec in CANDIDATES.items():
        train, valid = fold_batches[spec["r"]]
        theta, attempts = fit(train, config, spec, max_iterations)
        probs = terms(theta, valid)[0]
        predictions[name] = probs
        result = metrics(probs, valid)
        record["candidates"][name] = {"parameters": dict(zip(NAMES, theta.tolist())), "attempts": attempts, "validation": result}
        print(f"Fold {index + 1}/5 {name}: NLL={result['nll']:.6f}, accuracy={result['accuracy']:.4%}", flush=True)
    record["candidates"]["mixture"] = {"validation": metrics(0.5 * (predictions["baseline"] + predictions["history_full"]), fold_batches[200.0][1])}
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/public_train.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/final_development_cv.json"))
    parser.add_argument("--max-iterations", type=int, default=250)
    parser.add_argument("--refit", action="store_true", help="After CV, write a NEW final artifact; does not change config.")
    parser.add_argument("--parameter-output", type=Path, default=Path("artifacts/refitted_candidate.json"), help="Full-public refit output; defaults away from the deployed artifact.")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1, help="Independent fold processes (1-4); does not change seeds or candidates.")
    parser.add_argument("--resume", action="store_true", help="Reuse complete folds in the aggregate report after checking the protocol/input.")
    args = parser.parse_args()
    if args.refit and args.parameter_output.resolve() in {Path("artifacts/fitted_params.json").resolve(), Path("artifacts/fitted_params_final.json").resolve()}:
        parser.error("Refit output must not overwrite a preserved/deployed artifact")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    trajectories = load_trajectories(args.data)
    subjects = sorted({t["context"]["subject_id"] for t in trajectories})
    if len(subjects) < 5:
        raise ValueError("Five folds require >= 5 subjects")
    ids = {s: i for i, s in enumerate(subjects)}
    shuffled = list(range(len(subjects)))
    random.Random(SEED).shuffle(shuffled)
    folds = [np.asarray(shuffled[i::5]) for i in range(5)]
    report = {"schema_version": 1, "seed": SEED, "frozen_sha": "1f1899f80c65ad2523f153a8eacfd51eeccc449d", "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "protocol": "labbook/2026-08-27-final-development.md", "interpretation_limit": "Development CV; public data previously explored; intervals are not selection adjusted.",
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "subjects": len(subjects), "trajectories": len(trajectories), "max_iterations": args.max_iterations,
              "candidate_specifications": CANDIDATES, "folds": [], "pooled": {}}
    if args.resume and args.output.exists():
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        for key in ("data_sha256", "seed", "subjects", "trajectories", "max_iterations", "candidate_specifications", "environment"):
            if saved[key] != report[key]:
                raise ValueError(f"Cannot resume with different {key}")
        report["folds"] = saved["folds"]
    batches = {}
    for r in (200.0, 100.0):
        print(f"Extracting causal features r={r:g}", flush=True)
        batches[r] = extract(trajectories, config_for_feature_extraction(config, 25.0, r), ids)
    full = batches[200.0]
    predictions = {name: np.zeros((len(full.last), full.n_actions)) for name in [*CANDIDATES, "mixture"]}
    completed = {f["fold"] for f in report["folds"]}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_fold, i, valid_ids, subjects, batches, config, args.max_iterations) for i, valid_ids in enumerate(folds) if i not in completed]
        for job in as_completed(jobs):
            report["folds"].append(job.result())
            report["folds"].sort(key=lambda f: f["fold"])
            write_json(args.output, report)
    for fold_record in report["folds"]:
        valid_ids = folds[fold_record["fold"]]
        mask = np.isin(full.subject, valid_ids)
        valid_batches = {r: b.subset(valid_ids) for r, b in batches.items()}
        expected_hash = hashlib.sha256("\n".join(str(subjects[i]) for i in sorted(valid_ids)).encode()).hexdigest()
        assert fold_record["validation_membership_sha256"] == expected_hash
        for name, spec in CANDIDATES.items():
            theta = np.array([fold_record["candidates"][name]["parameters"][key] for key in NAMES])
            probabilities = terms(theta, valid_batches[spec["r"]])[0]
            restored = metrics(probabilities, valid_batches[spec["r"]])
            assert abs(restored["nll"] - fold_record["candidates"][name]["validation"]["nll"]) < 1e-12
            predictions[name][mask] = probabilities
        predictions["mixture"][mask] = 0.5 * (predictions["baseline"][mask] + predictions["history_full"][mask])
    base_metrics = metrics(predictions["baseline"], full)
    eligible = []
    for name, probabilities in predictions.items():
        result = metrics(probabilities, full)
        improved_folds = sum(f["candidates"][name]["validation"]["nll"] < f["candidates"]["baseline"]["validation"]["nll"] for f in report["folds"])
        qualifies = name == "baseline" or (result["nll"] < base_metrics["nll"] and result["accuracy"] >= base_metrics["accuracy"] and improved_folds >= 4)
        report["pooled"][name] = {"metrics": result, "improved_nll_folds": improved_folds, "qualifies": qualifies, "paired_subject_bootstrap": paired_bootstrap(probabilities, predictions["baseline"], full, len(subjects))}
        if qualifies:
            eligible.append(name)
        print(f"Pooled {name}: NLL={result['nll']:.6f}, accuracy={result['accuracy']:.4%}, eligible={qualifies}", flush=True)
    selected = min(eligible, key=lambda name: report["pooled"][name]["metrics"]["nll"])
    report["selected"] = selected
    write_json(args.output, report)
    if args.refit and selected != "baseline":
        if selected == "mixture":
            raise RuntimeError("Mixture won: export two members explicitly; no silent architecture substitution")
        spec = CANDIDATES[selected]
        theta, attempts = fit(batches[spec["r"]], config, spec, args.max_iterations)
        params = dict(zip(NAMES, theta.tolist()))
        params.update({key: config["model"][key] for key in ("initial_value", "initial_variance", "variance_ceiling")})
        params.update(process_variance=25.0, observation_variance=spec["r"])
        artifact = {"schema_version": 1, "model_name": "history_stay_switch_kalman", "parameters": params, "fit_seed": SEED,
                    "development_report": str(args.output.as_posix()), "selected_candidate": selected, "fit_attempts": attempts,
                    "data_sha256": report["data_sha256"], "final_public_fit_metrics": metrics(terms(theta, batches[spec["r"]])[0], batches[spec["r"]])}
        write_json(args.parameter_output, artifact)
        print(f"Saved NEW full-public fit: {artifact['final_public_fit_metrics']}", flush=True)
    print(f"Selected: {selected}. Config and frozen artifact are unchanged.", flush=True)


if __name__ == "__main__":
    main()
