"""Fit frozen Stay-Switch Kalman parameters on the public trajectories."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from scipy.optimize import minimize

from agent import Agent


PARAMETER_NAMES = (
    "value_sensitivity",
    "uncertainty_sensitivity",
    "stay_intercept",
    "stay_value_weight",
    "surprise_linear_weight",
    "surprise_quadratic_weight",
    "run_length_weight",
    "trial_index_weight",
    "lapse",
)

PARAMETER_BOUNDS = (
    (0.0, 10.0),
    (-5.0, 5.0),
    (-10.0, 10.0),
    (-5.0, 5.0),
    (-5.0, 5.0),
    (-5.0, 5.0),
    (-5.0, 5.0),
    (-5.0, 5.0),
    (0.0, 0.20),
)


@dataclass
class FeatureBatch:
    """Vectorized decision features from one trajectory collection."""

    values: np.ndarray
    uncertainties: np.ndarray
    last_indices: np.ndarray
    observed_indices: np.ndarray
    stay: np.ndarray
    surprise: np.ndarray
    log_run_length: np.ndarray
    log_trial_index: np.ndarray
    first_trials: int
    n_actions: int

    @property
    def later_trials(self) -> int:
        return int(self.stay.size)

    @property
    def total_trials(self) -> int:
        return self.first_trials + self.later_trials


def parse_float_grid(value: str) -> list[float]:
    """Parse a comma-separated, non-empty grid of finite floats."""
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(not math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated finite floats")
    return values


def load_trajectories(path: Path) -> list[dict[str, Any]]:
    """Load JSONL trajectories without retaining blank lines."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def split_by_subject(
    trajectories: list[dict[str, Any]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    """Create a deterministic subject-disjoint development split."""
    subject_ids = sorted(
        {trajectory["context"]["subject_id"] for trajectory in trajectories}
    )
    random.Random(seed).shuffle(subject_ids)
    training_count = min(
        len(subject_ids) - 1,
        max(1, round(len(subject_ids) * (1.0 - validation_fraction))),
    )
    training_subjects = set(subject_ids[:training_count])
    validation_subjects = set(subject_ids[training_count:])

    training = [
        trajectory
        for trajectory in trajectories
        if trajectory["context"]["subject_id"] in training_subjects
    ]
    validation = [
        trajectory
        for trajectory in trajectories
        if trajectory["context"]["subject_id"] in validation_subjects
    ]
    return (
        training,
        validation,
        sorted(training_subjects),
        sorted(validation_subjects),
    )


def config_for_feature_extraction(
    base_config: dict[str, Any],
    process_variance: float,
    observation_variance: float,
) -> dict[str, Any]:
    """Disable artifact loading and set candidate Kalman variances."""
    config = copy.deepcopy(base_config)
    model = config.setdefault("model", {})
    model["fitted_params_path"] = None
    model["process_variance"] = float(process_variance)
    model["observation_variance"] = float(observation_variance)
    return config


def extract_features(
    trajectories: Iterable[dict[str, Any]],
    config: dict[str, Any],
) -> FeatureBatch:
    """Run the causal Kalman filter and collect pre-choice features."""
    agent = Agent(config)
    value_rows: list[list[float]] = []
    uncertainty_rows: list[list[float]] = []
    last_indices: list[int] = []
    observed_indices: list[int] = []
    stay_labels: list[float] = []
    surprises: list[float] = []
    log_run_lengths: list[float] = []
    log_trial_indices: list[float] = []
    first_trials = 0
    expected_n_actions: int | None = None

    for trajectory in trajectories:
        context = trajectory["context"]
        agent.reset(context)
        action_to_index = {
            action: index for index, action in enumerate(agent.actions)
        }
        n_actions = len(agent.actions)
        if expected_n_actions is None:
            expected_n_actions = n_actions
        elif n_actions != expected_n_actions:
            raise ValueError(
                "One fitted parameter set requires a fixed number of actions"
            )

        for trial in trajectory["trials"]:
            observed_action = trial["action"]
            if observed_action not in action_to_index:
                raise ValueError(
                    f"Observed action {observed_action!r} is unavailable"
                )

            if agent.last_action is None:
                first_trials += 1
            else:
                values, uncertainties = agent._standardized_features()
                value_rows.append([values[action] for action in agent.actions])
                uncertainty_rows.append(
                    [uncertainties[action] for action in agent.actions]
                )
                last_indices.append(action_to_index[agent.last_action])
                observed_indices.append(action_to_index[observed_action])
                stay_labels.append(float(observed_action == agent.last_action))
                surprises.append(agent.last_surprise)
                log_run_lengths.append(math.log1p(agent.run_length))
                log_trial_indices.append(math.log1p(agent.trial_count))

            agent.update(
                action=observed_action,
                reward=trial["reward"],
                info=trial.get("info"),
            )

    if expected_n_actions is None:
        raise ValueError("No trajectories were provided")

    return FeatureBatch(
        values=np.asarray(value_rows, dtype=np.float64),
        uncertainties=np.asarray(uncertainty_rows, dtype=np.float64),
        last_indices=np.asarray(last_indices, dtype=np.int64),
        observed_indices=np.asarray(observed_indices, dtype=np.int64),
        stay=np.asarray(stay_labels, dtype=np.float64),
        surprise=np.asarray(surprises, dtype=np.float64),
        log_run_length=np.asarray(log_run_lengths, dtype=np.float64),
        log_trial_index=np.asarray(log_trial_indices, dtype=np.float64),
        first_trials=first_trials,
        n_actions=expected_n_actions,
    )


def _probability_terms(
    parameters: np.ndarray,
    batch: FeatureBatch,
) -> dict[str, np.ndarray]:
    """Compute vectorized probabilities and derivative intermediates."""
    (
        value_sensitivity,
        uncertainty_sensitivity,
        stay_intercept,
        stay_value_weight,
        surprise_linear_weight,
        surprise_quadratic_weight,
        run_length_weight,
        trial_index_weight,
        lapse,
    ) = parameters

    rows = np.arange(batch.later_trials)
    utilities = (
        value_sensitivity * batch.values
        + uncertainty_sensitivity * batch.uncertainties
    )
    utility_max = np.max(utilities, axis=1, keepdims=True)
    all_weights = np.exp(utilities - utility_max)
    all_probabilities = all_weights / np.sum(
        all_weights,
        axis=1,
        keepdims=True,
    )
    base_last = np.clip(
        all_probabilities[rows, batch.last_indices],
        1e-9,
        1.0 - 1e-9,
    )
    base_last_logit = np.log(base_last / (1.0 - base_last))

    switch_utilities = utilities.copy()
    switch_utilities[rows, batch.last_indices] = -np.inf
    switch_max = np.max(switch_utilities, axis=1, keepdims=True)
    switch_weights = np.exp(switch_utilities - switch_max)
    switch_weights[rows, batch.last_indices] = 0.0
    switch_probabilities = switch_weights / np.sum(
        switch_weights,
        axis=1,
        keepdims=True,
    )
    conditional_observed = switch_probabilities[
        rows,
        batch.observed_indices,
    ]

    gate_features = np.column_stack(
        (
            np.ones(batch.later_trials),
            base_last_logit,
            batch.surprise,
            batch.surprise**2,
            batch.log_run_length,
            batch.log_trial_index,
        )
    )
    gate_parameters = np.asarray(
        (
            stay_intercept,
            stay_value_weight,
            surprise_linear_weight,
            surprise_quadratic_weight,
            run_length_weight,
            trial_index_weight,
        )
    )
    gate_logit = np.clip(gate_features @ gate_parameters, -40.0, 40.0)
    stay_probability = 1.0 / (1.0 + np.exp(-gate_logit))

    raw_observed = np.where(
        batch.stay == 1.0,
        stay_probability,
        (1.0 - stay_probability) * conditional_observed,
    )
    observed_probability = (
        (1.0 - lapse) * raw_observed
        + lapse / batch.n_actions
    )
    observed_probability = np.clip(observed_probability, 1e-12, 1.0)

    return {
        "all_probabilities": all_probabilities,
        "switch_probabilities": switch_probabilities,
        "conditional_observed": conditional_observed,
        "gate_features": gate_features,
        "stay_probability": stay_probability,
        "raw_observed": raw_observed,
        "observed_probability": observed_probability,
    }


def objective_and_gradient(
    parameters: np.ndarray,
    batch: FeatureBatch,
    l2_penalty: float,
) -> tuple[float, np.ndarray]:
    """Return mean NLL and an analytic gradient for L-BFGS-B."""
    terms = _probability_terms(parameters, batch)
    stay_probability = terms["stay_probability"]
    conditional_observed = terms["conditional_observed"]
    raw_observed = terms["raw_observed"]
    observed_probability = terms["observed_probability"]
    switch_probabilities = terms["switch_probabilities"]
    gate_features = terms["gate_features"]

    lapse = parameters[-1]
    loss = -float(np.mean(np.log(observed_probability)))

    d_loss_d_raw = -(1.0 - lapse) / observed_probability
    d_raw_d_gate_logit = np.where(
        batch.stay == 1.0,
        stay_probability * (1.0 - stay_probability),
        -conditional_observed
        * stay_probability
        * (1.0 - stay_probability),
    )
    d_loss_d_gate_logit = d_loss_d_raw * d_raw_d_gate_logit

    gradient = np.zeros_like(parameters)
    gradient[2:8] = np.mean(
        d_loss_d_gate_logit[:, None] * gate_features,
        axis=0,
    )

    rows = np.arange(batch.later_trials)
    expected_switch_value = np.sum(
        switch_probabilities * batch.values,
        axis=1,
    )
    expected_switch_uncertainty = np.sum(
        switch_probabilities * batch.uncertainties,
        axis=1,
    )
    last_value = batch.values[rows, batch.last_indices]
    last_uncertainty = batch.uncertainties[
        rows,
        batch.last_indices,
    ]
    observed_value = batch.values[rows, batch.observed_indices]
    observed_uncertainty = batch.uncertainties[
        rows,
        batch.observed_indices,
    ]

    d_base_logit_d_value = last_value - expected_switch_value
    d_base_logit_d_uncertainty = (
        last_uncertainty - expected_switch_uncertainty
    )
    d_conditional_d_value = conditional_observed * (
        observed_value - expected_switch_value
    )
    d_conditional_d_uncertainty = conditional_observed * (
        observed_uncertainty - expected_switch_uncertainty
    )

    switch_indicator = 1.0 - batch.stay
    gate_value_term = (
        d_raw_d_gate_logit
        * parameters[3]
        * d_base_logit_d_value
    )
    gate_uncertainty_term = (
        d_raw_d_gate_logit
        * parameters[3]
        * d_base_logit_d_uncertainty
    )
    switch_value_term = (
        switch_indicator
        * (1.0 - stay_probability)
        * d_conditional_d_value
    )
    switch_uncertainty_term = (
        switch_indicator
        * (1.0 - stay_probability)
        * d_conditional_d_uncertainty
    )

    gradient[0] = np.mean(
        d_loss_d_raw * (gate_value_term + switch_value_term)
    )
    gradient[1] = np.mean(
        d_loss_d_raw
        * (gate_uncertainty_term + switch_uncertainty_term)
    )
    gradient[-1] = np.mean(
        (raw_observed - 1.0 / batch.n_actions)
        / observed_probability
    )

    regularized = parameters[2:8]
    loss += l2_penalty * float(np.sum(regularized**2))
    gradient[2:8] += 2.0 * l2_penalty * regularized
    return loss, gradient


def fit_parameters(
    batch: FeatureBatch,
    initial_parameters: np.ndarray,
    max_iterations: int,
    l2_penalty: float,
) -> np.ndarray:
    """Fit global decision parameters with bounded maximum likelihood."""
    result = minimize(
        objective_and_gradient,
        x0=initial_parameters,
        args=(batch, l2_penalty),
        method="L-BFGS-B",
        jac=True,
        bounds=PARAMETER_BOUNDS,
        options={
            "maxiter": max_iterations,
            "ftol": 1e-11,
            "gtol": 1e-7,
            "maxls": 30,
        },
    )
    if not result.success:
        print(f"Warning: optimizer stopped with: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def evaluate_parameters(
    parameters: np.ndarray,
    batch: FeatureBatch,
) -> dict[str, float | int]:
    """Compute official-style aggregate and stay/switch diagnostics."""
    terms = _probability_terms(parameters, batch)
    observed_probability = terms["observed_probability"]
    later_losses = -np.log(observed_probability)
    first_loss = math.log(batch.n_actions)

    stay_mask = batch.stay == 1.0
    switch_mask = ~stay_mask

    stay_probability = terms["stay_probability"]
    switch_probabilities = terms["switch_probabilities"]
    later_probabilities = (
        (1.0 - stay_probability)[:, None] * switch_probabilities
    )
    rows = np.arange(batch.later_trials)
    later_probabilities[rows, batch.last_indices] = stay_probability
    later_probabilities = (
        (1.0 - parameters[-1]) * later_probabilities
        + parameters[-1] / batch.n_actions
    )
    maximum_probability = np.max(
        later_probabilities,
        axis=1,
        keepdims=True,
    )
    winners = later_probabilities == maximum_probability
    observed_is_winner = winners[rows, batch.observed_indices]
    later_credit = observed_is_winner / np.sum(winners, axis=1)
    later_correct = np.sum(later_credit)

    total_loss = float(np.sum(later_losses) + batch.first_trials * first_loss)
    total_accuracy = float(
        later_correct + batch.first_trials / batch.n_actions
    )
    return {
        "nll": total_loss / batch.total_trials,
        "accuracy": total_accuracy / batch.total_trials,
        "stay_nll": float(np.mean(later_losses[stay_mask])),
        "switch_nll": float(np.mean(later_losses[switch_mask])),
        "first_trials": batch.first_trials,
        "later_trials": batch.later_trials,
        "total_trials": batch.total_trials,
    }


def parameters_from_config(config: dict[str, Any]) -> np.ndarray:
    """Construct an optimizer start vector from the YAML configuration."""
    model = config["model"]
    return np.asarray(
        [float(model[name]) for name in PARAMETER_NAMES],
        dtype=np.float64,
    )


def print_metrics(label: str, metrics: dict[str, float | int]) -> None:
    """Print a compact, comparable metric summary."""
    print(
        f"{label}: "
        f"NLL={metrics['nll']:.6f}, "
        f"accuracy={float(metrics['accuracy']):.4%}, "
        f"stay_NLL={metrics['stay_nll']:.6f}, "
        f"switch_NLL={metrics['switch_nll']:.6f}, "
        f"trials={metrics['total_trials']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the Stay-Switch Kalman model with a subject-disjoint "
            "validation split."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/public_train.jsonl"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/fitted_params.json"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--l2-penalty", type=float, default=1e-5)
    parser.add_argument(
        "--process-variance-grid",
        type=parse_float_grid,
        default=[25.0],
        help="Comma-separated candidates, for example 10,25,50.",
    )
    parser.add_argument(
        "--observation-variance-grid",
        type=parse_float_grid,
        default=[100.0],
        help="Comma-separated candidates, for example 50,100,200.",
    )
    args = parser.parse_args()

    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between zero and one")
    if args.l2_penalty < 0.0:
        parser.error("--l2-penalty cannot be negative")

    with args.config.open(encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream)
    trajectories = load_trajectories(args.data)
    training, validation, training_subjects, validation_subjects = (
        split_by_subject(
            trajectories,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    )
    print(
        f"Loaded {len(trajectories)} trajectories from "
        f"{len(training_subjects) + len(validation_subjects)} subjects."
    )
    print(
        f"Development split: {len(training_subjects)} training subjects, "
        f"{len(validation_subjects)} validation subjects."
    )

    initial_parameters = parameters_from_config(base_config)
    best: dict[str, Any] | None = None

    for process_variance in args.process_variance_grid:
        for observation_variance in args.observation_variance_grid:
            candidate_config = config_for_feature_extraction(
                base_config,
                process_variance,
                observation_variance,
            )
            training_batch = extract_features(training, candidate_config)
            validation_batch = extract_features(validation, candidate_config)
            fitted = fit_parameters(
                training_batch,
                initial_parameters,
                max_iterations=args.max_iterations,
                l2_penalty=args.l2_penalty,
            )
            training_metrics = evaluate_parameters(fitted, training_batch)
            validation_metrics = evaluate_parameters(fitted, validation_batch)

            label = (
                f"q={process_variance:g}, r={observation_variance:g}"
            )
            print_metrics(f"Training [{label}]", training_metrics)
            print_metrics(f"Validation [{label}]", validation_metrics)

            candidate = {
                "process_variance": float(process_variance),
                "observation_variance": float(observation_variance),
                "parameters": fitted,
                "training_metrics": training_metrics,
                "validation_metrics": validation_metrics,
            }
            if (
                best is None
                or float(validation_metrics["nll"])
                < float(best["validation_metrics"]["nll"])
            ):
                best = candidate

    if best is None:
        raise RuntimeError("No parameter candidate was evaluated")

    final_config = config_for_feature_extraction(
        base_config,
        best["process_variance"],
        best["observation_variance"],
    )
    full_batch = extract_features(trajectories, final_config)
    final_parameters = fit_parameters(
        full_batch,
        best["parameters"],
        max_iterations=args.max_iterations,
        l2_penalty=args.l2_penalty,
    )
    public_metrics = evaluate_parameters(final_parameters, full_batch)
    print_metrics("Final public fit", public_metrics)

    parameter_mapping = {
        name: float(value)
        for name, value in zip(PARAMETER_NAMES, final_parameters)
    }
    parameter_mapping.update(
        {
            "initial_value": float(
                base_config["model"]["initial_value"]
            ),
            "initial_variance": float(
                base_config["model"]["initial_variance"]
            ),
            "process_variance": float(best["process_variance"]),
            "observation_variance": float(
                best["observation_variance"]
            ),
            "variance_ceiling": float(
                base_config["model"]["variance_ceiling"]
            ),
        }
    )

    artifact = {
        "schema_version": 1,
        "model_name": "stay_switch_kalman_v0_2",
        "fit_seed": args.seed,
        "data": {
            "file": args.data.name,
            "trajectories": len(trajectories),
            "subjects": len(training_subjects) + len(validation_subjects),
            "trials": full_batch.total_trials,
        },
        "selection": {
            "split": "subject_disjoint",
            "validation_fraction": args.validation_fraction,
            "training_subjects": len(training_subjects),
            "validation_subjects": len(validation_subjects),
            "selected_process_variance": best["process_variance"],
            "selected_observation_variance": (
                best["observation_variance"]
            ),
            "training_metrics": best["training_metrics"],
            "validation_metrics": best["validation_metrics"],
        },
        "final_public_fit_metrics": public_metrics,
        "parameters": parameter_mapping,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(artifact, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Saved frozen parameters to {args.output}")


if __name__ == "__main__":
    main()
